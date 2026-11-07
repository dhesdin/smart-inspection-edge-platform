from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset

from smart_inspection.evaluation.scores import collect_scores
from smart_inspection.models.base import AnomalyMethod


class _FakeMethod(AnomalyMethod):
    """AnomalyMethod whose anomaly score is the mean pixel value of the image, and that counts predict calls."""

    def __init__(self) -> None:
        self.n_predict_calls = 0

    @property
    def requires_validation(self) -> bool:
        return False

    def save(self, file_path: Path) -> None:
        pass

    def load(self, file_path: Path) -> None:
        pass

    def fit(self, train_loader, val_loader=None) -> None:
        pass

    def predict(self, image: Tensor) -> tuple[float, Tensor]:
        self.n_predict_calls += 1
        return image.mean().item(), image[0]


class _ConstantImageDataset(Dataset):
    """Each sample is an image filled with a known value, so its expected score is that value.

    When tensor_labels is True, labels are returned as 0-d tensors (as a DataLoader-free Dataset may do).
    """

    def __init__(self, samples: list[tuple[float, int]], tensor_labels: bool = False) -> None:
        self.samples = samples
        self.tensor_labels = tensor_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        value, label = self.samples[idx]
        return {
            "image": torch.full((3, 4, 4), value),
            "label": torch.tensor(label) if self.tensor_labels else label,
        }


SAMPLES = [(0.1, 0), (0.5, 1), (0.9, 1)]


# Each scores is the mean of the constant image
def test_collect_scores_returns_aligned_scores_and_labels():
    scores, labels = collect_scores(_FakeMethod(), _ConstantImageDataset(SAMPLES))

    assert scores == pytest.approx([0.1, 0.5, 0.9])
    assert labels == [0, 1, 1]


# predict is called exactly once per image.
def test_collect_scores_calls_predict_once_per_image():
    method = _FakeMethod()

    collect_scores(method, _ConstantImageDataset(SAMPLES))

    assert method.n_predict_calls == 3


# Outputs are plain Python types (JSON friendly): float scores, int labels, even when labels are tensors.
@pytest.mark.parametrize("tensor_labels", [False, True])
def test_collect_scores_returns_python_types(tensor_labels):
    scores, labels = collect_scores(_FakeMethod(), _ConstantImageDataset(SAMPLES, tensor_labels=tensor_labels))

    assert all(type(s) is float for s in scores)
    assert all(type(label) is int for label in labels)


# A Subset is served in its own index order: indices [2, 0] give the scores of samples 2 then 0.
def test_collect_scores_follows_subset_order():
    subset = Subset(_ConstantImageDataset(SAMPLES), [2, 0])

    scores, labels = collect_scores(_FakeMethod(), subset)

    assert scores == pytest.approx([0.9, 0.1])
    assert labels == [1, 0]


# An empty dataset is rejected with a clear error, before any predict call.
def test_collect_scores_rejects_empty_dataset():
    method = _FakeMethod()

    with pytest.raises(ValueError, match="Dataset is empty"):
        collect_scores(method, _ConstantImageDataset([]))

    assert method.n_predict_calls == 0