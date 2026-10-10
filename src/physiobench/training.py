"""Small, genuinely trainable waveform models for the synthetic research benchmark.

Both models combine observable temporal features with a deterministic spectrum.
They do not receive simulator parameters, subject IDs, or scenario labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


TARGET_CENTER = np.array([100.0, 20.0], dtype=np.float32)
TARGET_SCALE = np.array([40.0, 10.0], dtype=np.float32)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 24
    batch_size: int = 64
    learning_rate: float = 0.002
    weight_decay: float = 0.0001
    corruption_weight: float = 0.15
    quality_weight: float = 0.05
    severity_weight: float = 0.05
    target_coverage: float = 0.80
    coverage_penalty: float = 8.0
    gradient_clip: float = 2.0
    cpu_threads: int = 2


def seed_everything(seed: int, threads: int = 2) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    # CPU kernels are used for the published study; unsupported deterministic
    # operations fail loudly rather than silently changing reproducibility.
    torch.use_deterministic_algorithms(True)


class ObservableFeatures(nn.Module):
    """Normalize each observed window and retain an explicit missingness mask.

    The 0.1--4 Hz spectrum is a disclosed physiological inductive bias, shared
    by both architectures. This is not an end-to-end foundation-model study.
    """
    def __init__(self, fs: float, n_samples: int, bins: int = 80) -> None:
        super().__init__()
        frequencies = torch.fft.rfftfreq(n_samples, d=1.0 / fs)
        self.register_buffer("band", (frequencies >= 0.1) & (frequencies <= 4.0))
        self.register_buffer("window", torch.hann_window(n_samples))
        self.bins = bins

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = observed.to(x.dtype)
        centered = x - x.mean(dim=-1, keepdim=True)
        scale = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
        normalized = (centered / scale).clamp(-10, 10) * mask[..., None]
        power = torch.fft.rfft(normalized * self.window, dim=-1).abs().square()[..., self.band]
        power = power / power.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        spectrum = F.adaptive_avg_pool1d(power, self.bins).flatten(start_dim=1)
        amplitude = scale.squeeze(-1).clamp_min(1e-5).log().clamp(-12, 8) * mask
        auxiliary = torch.cat([spectrum, amplitude, mask], dim=-1)
        return normalized, auxiliary


class PhysioModel(nn.Module):
    def __init__(self, architecture: str, fs: float, n_samples: int) -> None:
        super().__init__()
        if architecture not in {"cnn", "transformer"}:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.architecture = architecture
        self.preprocess = ObservableFeatures(fs, n_samples)
        self.spectral = nn.Sequential(nn.Linear(328, 96), nn.GELU(), nn.Linear(96, 48), nn.GELU())
        if architecture == "cnn":
            self.temporal = nn.Sequential(
                nn.Conv1d(4, 16, 15, stride=5, padding=7), nn.GELU(),
                nn.Conv1d(16, 32, 9, stride=5, padding=4), nn.GELU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            )
        else:
            self.patch = nn.Conv1d(4, 32, 20, stride=20)
            layer = nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, dropout=0.0,
                                              activation="gelu", batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.fusion = nn.Sequential(nn.Linear(80, 64), nn.GELU(), nn.Linear(64, 48), nn.GELU())
        self.regression = nn.Linear(48, 2)
        self.corruption = nn.Linear(48, 4)
        self.quality = nn.Linear(48, 1)
        self.severity = nn.Linear(48, 4)

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized, auxiliary = self.preprocess(x, observed)
        if self.architecture == "cnn":
            temporal = self.temporal(normalized)
        else:
            tokens = self.patch(normalized).transpose(1, 2)
            length, width = tokens.shape[1:]
            position = torch.arange(length, device=x.device, dtype=x.dtype)[:, None]
            rate = torch.exp(torch.arange(0, width, 2, device=x.device, dtype=x.dtype)
                             * (-np.log(10000.0) / width))
            encoding = torch.zeros((length, width), device=x.device, dtype=x.dtype)
            encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * rate), torch.cos(position * rate)
            temporal = self.temporal(tokens + encoding).mean(dim=1)
        fused = self.fusion(torch.cat([temporal, self.spectral(auxiliary)], dim=1))
        return {"regression": self.regression(fused), "corruption_logits": self.corruption(fused),
                "quality_logits": self.quality(fused).squeeze(-1),
                "severity_prediction": self.severity(fused).sigmoid()}


def tensor_data(dataset: Any) -> dict[str, torch.Tensor]:
    return {"x": torch.from_numpy(np.asarray(dataset.x, dtype=np.float32)),
            "observed": torch.from_numpy(np.asarray(dataset.observed, dtype=bool)),
            "y": torch.from_numpy((np.asarray(dataset.y, dtype=np.float32) - TARGET_CENTER) / TARGET_SCALE),
            "corruption": torch.from_numpy(np.asarray(dataset.corruption, dtype=np.float32)),
            "severity": torch.from_numpy(np.asarray(dataset.severity, dtype=np.float32))}


def quality_target(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = (prediction.detach() - target).abs()
    return ((residual[:, 0] <= 5.0 / TARGET_SCALE[0]) &
            (residual[:, 1] <= 3.0 / TARGET_SCALE[1])).to(prediction.dtype)


def multitask_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                   config: TrainConfig, selective: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
    risk = F.smooth_l1_loss(output["regression"], batch["y"], reduction="none").mean(dim=1)
    observed = batch["observed"].to(torch.float32)
    corruption = F.binary_cross_entropy_with_logits(output["corruption_logits"], batch["corruption"],
                                                     reduction="none")
    corruption = (corruption * observed).sum() / observed.sum().clamp_min(1)
    quality = F.binary_cross_entropy_with_logits(output["quality_logits"],
                                                 quality_target(output["regression"], batch["y"]))
    severity = (((output["severity_prediction"] - batch["severity"]) ** 2) * observed).sum() / observed.sum().clamp_min(1)
    loss = risk.mean() + config.corruption_weight * corruption + config.quality_weight * quality + config.severity_weight * severity
    coverage_penalty = torch.zeros_like(loss)
    if selective:
        acceptance = output["quality_logits"].sigmoid()
        selective_risk = (acceptance * risk).sum() / acceptance.sum().clamp_min(1e-4)
        coverage_penalty = config.coverage_penalty * F.relu(config.target_coverage - acceptance.mean()).square()
        # Preserve an ordinary regression objective: rejecting hard examples does
        # not eliminate pressure to learn their correct rates.
        loss = 0.5 * risk.mean() + 0.5 * selective_risk + coverage_penalty + \
               config.corruption_weight * corruption + config.quality_weight * quality + config.severity_weight * severity
    return loss, {"loss": float(loss.detach()), "regression": float(risk.mean().detach()),
                  "corruption": float(corruption.detach()), "quality": float(quality.detach()),
                  "severity": float(severity.detach()),
                  "coverage_penalty": float(coverage_penalty.detach())}


@torch.no_grad()
def predict(model: PhysioModel, dataset: Any, batch_size: int = 128) -> dict[str, np.ndarray]:
    model.eval()
    tensors = tensor_data(dataset)
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in ("regression", "corruption_logits", "quality_logits", "severity_prediction")}
    for start in range(0, len(dataset.y), batch_size):
        output = model(tensors["x"][start:start + batch_size], tensors["observed"][start:start + batch_size])
        for key, value in output.items():
            chunks[key].append(value.cpu().numpy())
    joined = {key: np.concatenate(value) for key, value in chunks.items()}
    joined["prediction"] = joined.pop("regression") * TARGET_SCALE + TARGET_CENTER
    return joined


def validation_score(model: PhysioModel, dataset: Any) -> float:
    prediction = predict(model, dataset)["prediction"]
    return float(np.abs((prediction - dataset.y) / TARGET_SCALE).mean())


def train(model: PhysioModel, training_data: Any, validation_data: Any, config: TrainConfig,
          seed: int, selective: bool = False) -> dict[str, Any]:
    """Fit using train subjects and choose an epoch using tuning subjects only."""
    seed_everything(seed, config.cpu_threads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    training = tensor_data(training_data)
    generator = torch.Generator().manual_seed(seed)
    best_score, best_state, best_epoch = np.inf, None, -1
    history = []
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(training_data.y), generator=generator)
        totals = []
        for indices in order.split(config.batch_size):
            batch = {key: value[indices] for key, value in training.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["x"], batch["observed"])
            loss, components = multitask_loss(output, batch, config, selective=selective)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            totals.append(components)
        score = validation_score(model, validation_data)
        history.append({"epoch": epoch + 1, "validation_normalized_mae": score,
                        **{key: float(np.mean([row[key] for row in totals])) for key in totals[0]}})
        if score < best_score:
            best_score, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch + 1
    assert best_state is not None
    model.load_state_dict(best_state)
    return {"configuration": asdict(config), "best_epoch": best_epoch, "best_validation_score": best_score,
            "history": history, "n_parameters": sum(p.numel() for p in model.parameters()),
            "selective_objective": selective}
