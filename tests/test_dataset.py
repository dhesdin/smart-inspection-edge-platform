from pathlib import Path

import pytest

from smart_inspection.config.loader import resolve_config_paths
from smart_inspection.data.dataset import AnomalyDataset


@pytest.fixture
def bottle_test_dir() -> Path:
    """Path to the real MVTec "bottle/test" directory used as fixture data."""
    config = resolve_config_paths(config_path="common.yaml")
    return config["data"]["datasets"] / "bottle" / "test"


@pytest.fixture
def bottle_dataset() -> AnomalyDataset:
    """AnomalyDataset instance built from the real MVTec "bottle/test" split."""
    return AnomalyDataset(category="bottle", split="test")


def test_discover_samples_count_matches_filesystem(bottle_test_dir, bottle_dataset):
    """The number of discovered samples should match an independent glob of the PNGs on disk."""
    expected_count = sum(1 for _ in bottle_test_dir.glob("*/*.png"))
    assert len(bottle_dataset) == expected_count


def test_labels_are_zero_for_good_and_one_for_defects(bottle_dataset):
    """Samples from the "good" folder should be labeled 0, all others labeled 1."""
    for sample in bottle_dataset.samples:
        expected_label = 0 if sample["images_path"].parent.name == "good" else 1
        assert sample["label"] == expected_label


def test_mask_path_is_none_only_for_good_samples(bottle_dataset):
    """mask_path should be None for "good" samples and set for defect samples."""
    for sample in bottle_dataset.samples:
        if sample["label"] == 0:
            assert sample["mask_path"] is None
        else:
            assert sample["mask_path"] is not None


def test_raises_value_error_when_no_samples_found(tmp_path, monkeypatch):
    """AnomalyDataset should raise ValueError when the split directory contains no samples."""
    category = "empty_category"
    split = "empty_split"
    (tmp_path / category / split).mkdir(parents=True)

    real_config = resolve_config_paths(config_path="common.yaml")
    fake_config = {**real_config, "data": {**real_config["data"], "datasets": tmp_path}}
    monkeypatch.setattr("smart_inspection.data.dataset.resolve_config_paths", lambda config_path: fake_config)

    with pytest.raises(ValueError):
        AnomalyDataset(category=category, split=split)


def test_getitem_returns_expected_keys_and_shapes(bottle_dataset):
    """__getitem__ should return image/label/mask with the expected shapes for both classes."""
    good_idx = next(i for i, s in enumerate(bottle_dataset.samples) if s["label"] == 0)
    defect_idx = next(i for i, s in enumerate(bottle_dataset.samples) if s["label"] == 1)

    for idx in (good_idx, defect_idx):
        item = bottle_dataset[idx]
        assert set(item.keys()) == {"image", "label", "mask"}
        assert item["image"].shape == (3, *bottle_dataset.input_size)
        assert isinstance(item["label"], int)
        assert item["mask"].shape == (1, *bottle_dataset.input_size)
