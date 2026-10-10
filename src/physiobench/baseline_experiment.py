"""Frozen signal and observable-feature baselines under the neural-study protocol.

Run with the same configuration as the neural experiment. Models are trained on
development subjects only; all stresses of each held-out case remain paired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .baselines import SignalBaseline, waveform_features
from .data import generate_dataset, split_subjects
from .experiments import TEST_CONDITIONS, stress, mixture, split_validation, digest_arrays
from .metrics import evaluate_predictions, cluster_bootstrap_mean
from .posttraining import fit_subject_conformal, binary_metrics


RATE_SCALE = np.asarray([40.,10.])


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")


def interval_report(calibration: dict, cases, test_prediction: np.ndarray) -> dict:
    """Report subject-max conformal bands using the neural-study definition."""
    error = np.abs(cases.y-test_prediction)
    result = {}
    for level,specification in calibration['levels'].items():
        width = specification['half_width_bpm']
        if width is None:
            result[level] = None
            continue
        covered = error <= np.asarray(width)
        subject = np.stack([covered[cases.subject_id==identifier].all(axis=0)
                            for identifier in np.unique(cases.subject_id)]).mean(axis=0)
        result[level] = {'case_coverage_hr':float(covered[:,0].mean()),'case_coverage_rr':float(covered[:,1].mean()),
                         'subject_coverage_hr':float(subject[0]),'subject_coverage_rr':float(subject[1]),
                         'width_hr_bpm':float(2*width[0]),'width_rr_bpm':float(2*width[1])}
    return result


def _classifier_probabilities(model: RandomForestClassifier, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    result = np.zeros((len(features),4),dtype=np.float32)
    for channel, (classes, probability) in enumerate(zip(model.classes_,probabilities)):
        positives = np.flatnonzero(classes==1)
        if len(positives):
            result[:,channel] = probability[:,positives[0]]
    return result


def run_baselines(configuration: dict, output: Path) -> dict:
    started = time.time()
    output.mkdir(parents=True,exist_ok=True)
    dataset = generate_dataset(**configuration['dataset'])
    splits = split_subjects(dataset,seed=configuration['split_seed'])
    _,calibration = split_validation(splits['val'],configuration['calibration_split_seed'])
    augmentation = tuple(condition for condition in TEST_CONDITIONS if not condition.startswith('drop_'))
    train = mixture(splits['train'],augmentation,configuration['stress_seed'])
    calibration_mix = mixture(calibration,augmentation,configuration['stress_seed'])
    test_data = {(split,condition):stress(splits[split],condition,configuration['stress_seed'])
                 for split in ('test_id','test_ood') for condition in TEST_CONDITIONS}
    print("Extracting observable-only baseline features",flush=True)
    train_features = waveform_features(train)
    calibration_features = waveform_features(calibration_mix)
    test_features = {key:waveform_features(part) for key,part in test_data.items()}
    records,comparisons,predictions = [],[],{}

    def evaluate(name,seed,calibration_prediction,predictor):
        intervals = fit_subject_conformal(calibration_prediction,calibration_mix.y,calibration_mix.subject_id)
        for (split,condition),cases in test_data.items():
            prediction = predictor(cases,test_features[(split,condition)])
            metrics = evaluate_predictions(cases,mean=prediction['mean'],
                                           corruption_prob=prediction['corruption_prob'],
                                           severity_pred=prediction['severity_pred'])
            metrics['regression_intervals'] = interval_report(intervals,cases,prediction['mean'])
            metrics['severity_mae_available'] = float(np.abs(prediction['severity_pred']-cases.severity)[cases.observed].mean())
            p = np.clip(prediction['corruption_prob'],1e-7,1-1e-7)
            logits = np.log(p)-np.log1p(-p)
            metrics['corruption_available'] = binary_metrics(logits,cases.corruption,mask=cases.observed)
            for j,target in enumerate(('hr','rr')):
                metrics[f'{target}_mae_subject_ci'] = cluster_bootstrap_mean(
                    np.abs(prediction['mean'][:,j]-cases.y[:,j]),cases.subject_id,
                    configuration['bootstrap_draws'],configuration['bootstrap_seed'])
            records.append({'model':name,'seed':seed,'split':split,'condition':condition,'metrics':metrics,
                            'interval_calibration':intervals})
            predictions[(name,seed,split,condition)] = prediction['mean']
            path = output/'predictions'/f'{name}-{seed}-{split}-{condition}.npz'
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**prediction,y=cases.y,case_id=cases.case_id,subject_id=cases.subject_id,
                                corruption=cases.corruption,severity=cases.severity,observed=cases.observed)

    signal = SignalBaseline().fit(splits['train'])
    evaluate('signal',0,signal.predict(calibration_mix)['mean'],lambda cases,features:signal.predict(cases))
    rf_configs = {'n_estimators':128,'max_depth':16,'min_samples_leaf':2,'n_jobs':2}
    for seed in configuration['seeds']:
        print(f"Fitting observable-feature random forest seed={seed}",flush=True)
        regression = RandomForestRegressor(**rf_configs,random_state=seed)
        # Normalize tasks so the multioutput split criterion does not let HR units
        # dominate severity. These fixed scales were not derived from test data.
        supervision = np.column_stack([train.y/RATE_SCALE,train.severity])
        regression.fit(train_features,supervision)
        detector = RandomForestClassifier(**rf_configs,random_state=seed)
        detector.fit(train_features,train.corruption.astype(int))
        cal_prediction = regression.predict(calibration_features)[:,:2]*RATE_SCALE
        def predict_feature(cases,features):
            output = regression.predict(features)
            return {'mean':output[:,:2]*RATE_SCALE,
                    'severity_pred':np.clip(output[:,2:],0,1),
                    'corruption_prob':_classifier_probabilities(detector,features)}
        evaluate('random_forest_70',seed,cal_prediction,predict_feature)
    # Pair the two predictions on identical cases and average RF seed differences
    # *within subject* before cluster resampling. Avoid an iid-window interval.
    for (split,condition),cases in test_data.items():
        signal_prediction = predictions[('signal',0,split,condition)]
        deltas = np.stack([np.abs(predictions[('random_forest_70',seed,split,condition)]-cases.y)
                           -np.abs(signal_prediction-cases.y) for seed in configuration['seeds']]).mean(axis=0)
        for j,target in enumerate(('hr','rr')):
            comparisons.append({'split':split,'condition':condition,'target':target,
                                'comparison':'random_forest_70_minus_signal',
                                **cluster_bootstrap_mean(deltas[:,j],cases.subject_id,
                                                         configuration['bootstrap_draws'],configuration['bootstrap_seed'])})
    _write(output/'metrics.json',records)
    _write(output/'paired_comparisons.json',comparisons)
    manifest = {'status':'completed','data_sha256':digest_arrays(dataset),'configuration':configuration,
                'feature_count':70,'random_forest_configuration':rf_configs,'rate_scale':RATE_SCALE.tolist(),
                'n_models':1+len(configuration['seeds']),'duration_s':time.time()-started,
                'splits':{name:np.unique(part.subject_id).tolist() for name,part in splits.items()},
                'calibration_subjects':np.unique(calibration.subject_id).tolist(),
                'software':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
                            'scikit_learn':sklearn.__version__},'hardware':platform.platform(),
                'limitations':['Entire primary benchmark is synthetic; no clinical validation',
                               'RF receives only 70 signal/mask features; metadata excluded',
                               'RF learned severity is simulator-control amplitude, not disease severity',
                               'Signal severity is a disclosed heuristic proxy from its corruption score',
                               'Subject-max conformal premise applies to new exchangeable synthetic ID subjects only',
                               'RF includes masks so missing-modality detection is a trivial observed-state task',
                               'CIs condition on simulator and fixed seeds; pointwise, not multiplicity-adjusted']}
    manifest['source_sha256'] = {path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(Path(__file__).parent.glob('*.py'))}
    _write(output/'configuration.json',configuration)
    manifest['artifacts_sha256'] = {str(path.relative_to(output)):hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(output.rglob('*')) if path.is_file() and path.name!='manifest.json'}
    _write(output/'manifest.json',manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/experiment-study.json'))
    parser.add_argument('--output',type=Path,default=Path('results/baselines'))
    args = parser.parse_args()
    result = run_baselines(json.loads(args.config.read_text()),args.output)
    print(json.dumps({'status':result['status'],'duration_s':result['duration_s'],'n_models':result['n_models']}),flush=True)


if __name__=='__main__':
    main()
