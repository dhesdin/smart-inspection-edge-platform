from pathlib import Path

import pytest
import torch
from PIL import Image

from smart_inspection.config.loader import resolve_config_paths
from smart_inspection.data.dataset import IMAGENET_MEAN, IMAGENET_STD, AnomalyDataset


@pytest.fixture
def bottle_test_dir() -> Path:
    """Path to the real MVTec "bottle/test" directory used as fixture data."""
    config = resolve_config_paths(config_path="common.yaml")
    return config["data"]["datasets"] / "bottle" / "test"


@pytest.fixture
def bottle_dataset() -> AnomalyDataset:
    """AnomalyDataset instance built from the real MVTec "bottle/test" split."""
    return AnomalyDataset(category="bottle", split="test")


@pytest.mark.integration
def test_discover_samples_count_matches_filesystem(bottle_test_dir, bottle_dataset):
    """The number of discovered samples should match an independent glob of the PNGs on disk."""
    expected_count = sum(1 for _ in bottle_test_dir.glob("*/*.png"))
    assert len(bottle_dataset) == expected_count


@pytest.mark.integration
def test_labels_are_zero_for_good_and_one_for_defects(bottle_dataset):
    """Samples from the "good" folder should be labeled 0, all others labeled 1."""
    for sample in bottle_dataset.samples:
        expected_label = 0 if sample["images_path"].parent.name == "good" else 1
        assert sample["label"] == expected_label


@pytest.mark.integration
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


@pytest.mark.integration
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


# ---------------- synthetic dataset (no MVTec needed, runs in CI) ---------------- #


@pytest.fixture
def synthetic_dataset(tmp_path, monkeypatch) -> AnomalyDataset:
    """MVTec-like "cat/test" tree built in tmp_path, deliberately created out of alphabetical order.

    Layout: good/ (RGB + grayscale images), defect_b/ and defect_a/ (RGB images with a half black / half white mask).
    """
    root = tmp_path / "cat"

    def save_image(path: Path, mode: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, (64, 64), color=128 if mode == "L" else (10, 120, 200)).save(path)

    def save_mask(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mask = Image.new("L", (64, 64), color=0)
        mask.paste(255, box=(32, 0, 64, 64))  # right half is the defect
        mask.save(path)

    for defect in ("defect_b", "defect_a"):  # b before a on purpose
        for name in ("2", "0", "1"):  # unsorted on purpose
            save_image(root / "test" / defect / f"{name}.png", "RGB")
            save_mask(root / "ground_truth" / defect / f"{name}_mask.png")
    save_image(root / "test" / "good" / "1.png", "L")  # grayscale, like MVTec grid/screw/zipper
    save_image(root / "test" / "good" / "0.png", "RGB")

    real_config = resolve_config_paths(config_path="common.yaml")
    fake_config = {**real_config, "data": {**real_config["data"], "datasets": tmp_path}}
    monkeypatch.setattr("smart_inspection.data.dataset.resolve_config_paths", lambda config_path: fake_config)

    return AnomalyDataset(category="cat", split="test")


# Samples are discovered in a deterministic order (sorted by folder, then by file name), whatever the filesystem returns.
def test_samples_are_sorted_by_folder_then_filename(synthetic_dataset):
    paths = [sample["images_path"] for sample in synthetic_dataset.samples]

    assert paths == sorted(paths)
    assert [p.parent.name for p in paths] == ["defect_a"] * 3 + ["defect_b"] * 3 + ["good"] * 2


# Two datasets built on the same files list their samples in the same order (needed for seeded splits to be reproducible).
def test_samples_order_is_identical_between_two_builds(synthetic_dataset):
    second_build = AnomalyDataset(category="cat", split="test")

    assert synthetic_dataset.samples == second_build.samples


# A grayscale image (mode "L", 1 channel) is converted to 3 channels so ImageNet normalization and ResNet can be applied.
def test_grayscale_image_is_returned_with_three_channels(synthetic_dataset):
    grayscale_idx = next(
        i for i, s in enumerate(synthetic_dataset.samples) if s["images_path"].parent.name == "good" and s["images_path"].name == "1.png"
    )

    item = synthetic_dataset[grayscale_idx]

    assert item["image"].shape == (3, *synthetic_dataset.input_size)


# A grayscale image gives the same value on the three channels (the gray channel is duplicated, not altered).
def test_grayscale_image_channels_are_identical(synthetic_dataset):
    grayscale_idx = next(i for i, s in enumerate(synthetic_dataset.samples) if s["images_path"].name == "1.png" and s["label"] == 0)

    image = synthetic_dataset[grayscale_idx]["image"]

    # the three channels are normalized with different ImageNet mean/std, so undo it before comparing
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    raw = image * std + mean
    assert torch.allclose(raw[0], raw[1], atol=1e-5)
    assert torch.allclose(raw[0], raw[2], atol=1e-5)


# An RGB image keeps 3 channels after the conversion.
def test_rgb_image_is_returned_with_three_channels(synthetic_dataset):
    rgb_idx = next(i for i, s in enumerate(synthetic_dataset.samples) if s["label"] == 1)

    item = synthetic_dataset[rgb_idx]

    assert item["image"].shape == (3, *synthetic_dataset.input_size)


# After resizing, the mask still contains only 0 and 1 (nearest interpolation, no blurred values on the defect border).
def test_mask_stays_binary_after_resize(synthetic_dataset):
    defect_idx = next(i for i, s in enumerate(synthetic_dataset.samples) if s["label"] == 1)

    mask = synthetic_dataset[defect_idx]["mask"]

    assert mask.shape == (1, *synthetic_dataset.input_size)
    assert set(mask.unique().tolist()) == {0.0, 1.0}
