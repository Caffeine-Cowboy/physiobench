import numpy as np
from sklearn.ensemble import RandomForestClassifier
from physiobench.baseline_experiment import _classifier_probabilities, interval_report
from physiobench.data import generate_dataset
from physiobench.posttraining import fit_subject_conformal


def test_multitask_classifier_respects_absent_positive_class():
    features = np.arange(8,dtype=float).reshape(-1,1)
    labels = np.zeros((8,4),dtype=int)
    labels[4:,0] = 1
    model = RandomForestClassifier(n_estimators=5,random_state=2).fit(features,labels)
    probabilities = _classifier_probabilities(model,features)
    assert probabilities.shape == (8,4)
    assert (probabilities[:,1:]==0).all()
    assert probabilities[-1,0] > probabilities[0,0]


def test_interval_report_measures_subject_collection_coverage():
    data = generate_dataset(n_subjects=20).subset(np.arange(6))
    prediction = data.y.copy()
    prediction[0,0] += 10
    calibration = {'levels':{'0.9':{'half_width_bpm':[5,1]},'0.95':{'half_width_bpm':None}}}
    report = interval_report(calibration,data,prediction)
    assert report['0.9']['case_coverage_hr'] == 5/6
    assert report['0.9']['subject_coverage_hr'] == 0
    assert report['0.9']['width_hr_bpm'] == 10
    assert report['0.95'] is None
