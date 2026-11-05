from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from smart_inspection.models.base import AnomalyMethod
from smart_inspection.training.train import train

SEED = 42


class _RecordingMethod(AnomalyMethod):
    """AnomalyMethod that records the loaders given to fit and the path given to save."""

    def __init__(self, requires_validation: bool = False) -> None:
        self._requires_validation = requires_validation
        self.n_fit_calls = 0
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.saved_path: Path | None = None

    @property
    def requires_validation(self) -> bool:
        return self._requires_validation

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


# fit is called exactly once, with the loaders built by train.
def test_train_calls_fit_once():
    method = _RecordingMethod()

    train(method, _CountingDataset(10), None, batch_size=2, seed=SEED)

    assert method.n_fit_calls == 1


# The datasets are used as given, without any further split.
def test_train_uses_datasets_as_given():
    method = _RecordingMethod(requires_validation=True)

    train(method, _CountingDataset(8), _CountingDataset(2), batch_size=2, seed=SEED)

    assert len(method.train_loader.dataset) == 8
    assert len(method.val_loader.dataset) == 2


# The batch size is forwarded to the loaders.
def test_train_uses_given_batch_size():
    method = _RecordingMethod(requires_validation=True)

    train(method, _CountingDataset(16), _CountingDataset(4), batch_size=4, seed=SEED)

    assert method.train_loader.batch_size == 4
    assert method.val_loader.batch_size == 4


# drop_last applies to the training loader only: the validation loader keeps every sample.
def test_train_drop_last_applies_to_training_loader_only():
    method = _RecordingMethod(requires_validation=True)

    train(method, _CountingDataset(7), _CountingDataset(3), batch_size=2, seed=SEED, drop_last=True)

    assert len(method.train_loader) == 3  # 7 // 2, the incomplete batch is dropped
    assert len(method.val_loader) == 2  # ceil(3 / 2), the incomplete batch is kept


# A training loader with zero batches (3 samples, batch_size=10, drop_last) is rejected before fit.
def test_train_rejects_empty_training_loader():
    method = _RecordingMethod()

    with pytest.raises(ValueError, match="Empty training loader"):
        train(method, _CountingDataset(3), None, batch_size=10, seed=SEED, drop_last=True)

    assert method.n_fit_calls == 0


# A method that requires validation cannot be trained without a validation dataset.
def test_train_rejects_missing_val_dataset_when_required():
    method = _RecordingMethod(requires_validation=True)

    with pytest.raises(ValueError, match="val_dataset is required"):
        train(method, _CountingDataset(10), None, batch_size=2, seed=SEED)

    assert method.n_fit_calls == 0


# A method that does not require validation ignores a provided val_dataset.
def test_train_ignores_val_dataset_when_not_required():
    method = _RecordingMethod(requires_validation=False)

    train(method, _CountingDataset(10), _CountingDataset(2), batch_size=2, seed=SEED)

    assert method.n_fit_calls == 1
    assert method.val_loader is None


# The seed controls the shuffling order of the training loader.
def test_train_seed_controls_shuffle_order():
    def first_epoch_order(seed: int) -> list[float]:
        method = _RecordingMethod()
        train(method, _CountingDataset(50), None, batch_size=5, seed=seed)
        return [v for batch in method.train_loader for v in batch["image"].flatten().tolist()]

    assert first_epoch_order(0) == first_epoch_order(0)
    assert first_epoch_order(0) != first_epoch_order(1)


# When save_path is given, the trained method is saved to that path.
def test_train_saves_model_when_save_path_is_given(tmp_path):
    method = _RecordingMethod()
    save_path = tmp_path / "model.pt"

    train(method, _CountingDataset(10), None, batch_size=2, seed=SEED, save_path=save_path)

    assert method.saved_path == save_path


# When save_path is None (default), nothing is saved.
def test_train_does_not_save_without_save_path():
    method = _RecordingMethod()

    train(method, _CountingDataset(10), None, batch_size=2, seed=SEED)

    assert method.saved_path is None
