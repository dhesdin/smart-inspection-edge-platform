from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from smart_inspection.evaluation.evaluate import evaluate
from smart_inspection.models.base import AnomalyMethod


class _FakeMethod(AnomalyMethod):
    """AnomalyMethod whose anomaly score is the mean pixel value of the image, and that counts predict calls."""

    def __init__(self) -> None:
        self.n_predict_calls = 0

    @property
    def validation_split_ratio(self) -> float | None:
        return None

    def save(self, file_path: Path) -> None:
        pass

    def load(self, file_path: Path) -> None:
        pass

    def fit(self, train_loader, val_loader=None) -> None:
        pass

    def predict(self, image: Tensor) -> tuple[float, Tensor]:
        self.n_predict_calls += 1
        return image.mean().item(), image[0]


class _WrongSizeMapMethod(_FakeMethod):
    """AnomalyMethod whose anomaly map is smaller than the mask (2x2 instead of 4x4)."""

    def predict(self, image: Tensor) -> tuple[float, Tensor]:
        return image.mean().item(), image[0, :2, :2]


class _ScoredDataset(Dataset):
    """Dataset of constant images: each sample is (score, label), the image is filled with its score."""

    def __init__(self, samples: list[tuple[float, int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        score, label = self.samples[idx]
        return {"image": torch.full((3, 4, 4), score), "label": label, "mask": torch.full((1, 4, 4), float(label))}


def _loader(samples: list[tuple[float, int]], batch_size: int = 2) -> DataLoader:
    return DataLoader(_ScoredDataset(samples), batch_size=batch_size, shuffle=False)


# A score that ranks every anomaly above every normal sample gives an AUROC of 1.0.
def test_evaluate_perfect_separation_gives_auroc_one():
    loader = _loader([(0.1, 0), (0.2, 0), (0.8, 1), (0.9, 1)])

    result = evaluate(_FakeMethod(), loader)

    assert result["roc_auc"] == pytest.approx(1.0)


# A score that ranks every anomaly below every normal sample gives an AUROC of 0.0.
def test_evaluate_inverted_separation_gives_auroc_zero():
    loader = _loader([(0.8, 0), (0.9, 0), (0.1, 1), (0.2, 1)])

    result = evaluate(_FakeMethod(), loader)

    assert result["roc_auc"] == pytest.approx(0.0)


# Identical scores for every sample carry no information: AUROC is 0.5.
def test_evaluate_constant_scores_give_auroc_half():
    loader = _loader([(0.5, 0), (0.5, 0), (0.5, 1), (0.5, 1)])

    result = evaluate(_FakeMethod(), loader)

    assert result["roc_auc"] == pytest.approx(0.5)


# The sample counts (total, normal, anomaly) are reported correctly.
def test_evaluate_reports_sample_counts():
    loader = _loader([(0.1, 0), (0.2, 0), (0.3, 0), (0.8, 1), (0.9, 1)])

    result = evaluate(_FakeMethod(), loader)

    assert result["n_samples"] == 5
    assert result["n_normal"] == 3
    assert result["n_anomaly"] == 2


# predict is called once per sample, including the last incomplete batch (5 samples, batch_size=2).
def test_evaluate_calls_predict_once_per_sample():
    method = _FakeMethod()
    loader = _loader([(0.1, 0), (0.2, 0), (0.3, 0), (0.8, 1), (0.9, 1)], batch_size=2)

    evaluate(method, loader)

    assert method.n_predict_calls == 5


# The AUROC is a plain Python float (JSON friendly), not a numpy scalar.
def test_evaluate_auroc_is_a_python_float():
    loader = _loader([(0.1, 0), (0.9, 1)])

    result = evaluate(_FakeMethod(), loader)

    assert type(result["roc_auc"]) is float


# A test set with only normal samples cannot give an AUROC: ValueError.
def test_evaluate_single_class_raises_value_error():
    loader = _loader([(0.1, 0), (0.2, 0)])

    with pytest.raises(ValueError, match="at least one normal and one anomalous"):
        evaluate(_FakeMethod(), loader)


# An empty loader is rejected with a clear error.
def test_evaluate_empty_loader_raises_value_error():
    loader = _loader([])

    with pytest.raises(ValueError, match="no samples"):
        evaluate(_FakeMethod(), loader)


# The pixel AUROC is a plain Python float (JSON friendly), not a numpy scalar.
def test_evaluate_pixel_auroc_is_a_python_float():
    loader = _loader([(0.1, 0), (0.9, 1)])

    result = evaluate(_FakeMethod(), loader)

    assert type(result["roc_auc_pixel"]) is float


# The pixel counts are reported correctly: 2 normal + 1 anomalous 4x4 images give 32 normal and 16 anomalous pixels.
def test_evaluate_reports_pixel_counts():
    loader = _loader([(0.1, 0), (0.2, 0), (0.9, 1)])

    result = evaluate(_FakeMethod(), loader)

    assert result["n_pixel_normal"] == 32
    assert result["n_pixel_anomaly"] == 16


# An anomaly map whose size differs from the mask is rejected with a clear error.
def test_evaluate_map_mask_size_mismatch_raises_value_error():
    loader = _loader([(0.1, 0), (0.9, 1)])

    with pytest.raises(ValueError, match="does not match mask"):
        evaluate(_WrongSizeMapMethod(), loader)
