from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from smart_inspection.models.base import AnomalyMethod
from smart_inspection.training.train import train


class _RecordingMethod(AnomalyMethod):
    """AnomalyMethod that records the loaders given to fit and the path given to save."""

    def __init__(self, validation_split_ratio: float | None = None) -> None:
        self._validation_split_ratio = validation_split_ratio
        self.n_fit_calls = 0
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.saved_path: Path | None = None

    @property
    def validation_split_ratio(self) -> float | None:
        return self._validation_split_ratio

    def save(self, file_path: Path) -> None:
        self.saved_path = file_path

    def load(self, file_path: Path) -> None:
        pass

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        self.n_fit_calls += 1
        self.train_loader = train_loader
        self.val_loader = val_loader

    def predict(self, image):
        return 0.0, image


class _CountingDataset(Dataset):
    """Dataset of n samples; sample i holds the value i."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {"image": torch.tensor([float(idx)]), "label": 0}


# Without validation split (ratio None), the whole dataset is used for training and no val loader is built.
def test_train_without_validation_split_uses_whole_dataset():
    method = _RecordingMethod(validation_split_ratio=None)

    train(method, _CountingDataset(10), batch_size=2)

    assert len(method.train_loader.dataset) == 10
    assert method.val_loader is None


# With a validation ratio, the dataset is split into train and validation subsets of the right sizes.
def test_train_with_validation_split_splits_the_dataset():
    method = _RecordingMethod(validation_split_ratio=0.2)

    train(method, _CountingDataset(10), batch_size=2)

    assert len(method.train_loader.dataset) == 8
    assert len(method.val_loader.dataset) == 2


# fit is called exactly once, with the loaders built by train.
def test_train_calls_fit_once():
    method = _RecordingMethod()

    train(method, _CountingDataset(10), batch_size=2)

    assert method.n_fit_calls == 1


# The batch size is forwarded to the loaders.
def test_train_uses_given_batch_size():
    method = _RecordingMethod(validation_split_ratio=0.2)

    train(method, _CountingDataset(20), batch_size=4)

    assert method.train_loader.batch_size == 4
    assert method.val_loader.batch_size == 4


# drop_last applies to the training loader only: the validation loader keeps every sample.
def test_train_drop_last_applies_to_training_loader_only():
    method = _RecordingMethod(validation_split_ratio=0.3)

    # 10 samples -> 7 train / 3 val, batch_size=2
    train(method, _CountingDataset(10), batch_size=2, drop_last=True)

    assert len(method.train_loader) == 3  # 7 // 2, the incomplete batch is dropped
    assert len(method.val_loader) == 2  # ceil(3 / 2), the incomplete batch is kept


# The same seed gives the same train/validation split, a different seed gives a different one.
def test_train_split_is_reproducible_with_seed():
    def train_indices(seed: int) -> list[int]:
        method = _RecordingMethod(validation_split_ratio=0.2)
        train(method, _CountingDataset(100), batch_size=4, seed=seed)
        return list(method.train_loader.dataset.indices)

    assert train_indices(seed=0) == train_indices(seed=0)
    assert train_indices(seed=0) != train_indices(seed=1)


# A validation ratio outside the open interval (0, 1) is rejected.
@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
def test_train_rejects_invalid_validation_ratio(ratio):
    method = _RecordingMethod(validation_split_ratio=ratio)

    with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
        train(method, _CountingDataset(10), batch_size=2)


# A split that leaves the validation set empty (1 sample, ratio 0.5) is rejected before fit.
def test_train_rejects_split_with_empty_subset():
    method = _RecordingMethod(validation_split_ratio=0.5)

    with pytest.raises(ValueError, match="empty subset"):
        train(method, _CountingDataset(1), batch_size=1)

    assert method.n_fit_calls == 0


# A training loader with zero batches (3 samples, batch_size=10, drop_last) is rejected before fit.
def test_train_rejects_empty_training_loader():
    method = _RecordingMethod()

    with pytest.raises(ValueError, match="Empty training loader"):
        train(method, _CountingDataset(3), batch_size=10, drop_last=True)

    assert method.n_fit_calls == 0


# When save_path is given, the trained method is saved to that path.
def test_train_saves_model_when_save_path_is_given(tmp_path):
    method = _RecordingMethod()
    save_path = tmp_path / "model.pt"

    train(method, _CountingDataset(10), batch_size=2, save_path=save_path)

    assert method.saved_path == save_path


# When save_path is None (default), nothing is saved.
def test_train_does_not_save_without_save_path():
    method = _RecordingMethod()

    train(method, _CountingDataset(10), batch_size=2)

    assert method.saved_path is None
