import pytest

from smart_inspection.data.splits import split_data

CAL = 0.2
VAL = 0.2
SEED = 42


# With 224 images and default ratios: int(224 * 0.2) = 44, int(180 * 0.2) = 36, rest = 144.
def test_split_data_sizes_for_224_images():
    calibration, validation, training = split_data(224, CAL, VAL, SEED)

    assert len(calibration) == 44
    assert len(validation) == 36
    assert len(training) == 144


# The three sets must not share any index.
def test_split_data_sets_are_disjoint():
    calibration, validation, training = split_data(224, CAL, VAL, SEED)

    assert not set(calibration) & set(validation)
    assert not set(calibration) & set(training)
    assert not set(validation) & set(training)


# Together the three sets cover every index exactly once.
def test_split_data_covers_all_indices():
    calibration, validation, training = split_data(224, CAL, VAL, SEED)

    all_indices = calibration + validation + training
    assert len(all_indices) == 224
    assert sorted(all_indices) == list(range(224))


# Same seed gives the same split.
def test_split_data_same_seed_is_reproducible():
    assert split_data(224, CAL, VAL, 7) == split_data(224, CAL, VAL, 7)


# A different seed gives a different split.
def test_split_data_different_seed_gives_different_split():
    assert split_data(224, CAL, VAL, 1) != split_data(224, CAL, VAL, 2)


# Sizes are rounded down with int(), not rounded to nearest.
def test_split_data_rounds_down_with_int():
    calibration, validation, training = split_data(10, 0.25, 0.25, SEED)

    # int(2.5) = 2, remaining 8, int(2.0) = 2
    assert (len(calibration), len(validation), len(training)) == (2, 2, 6)


# Zero ratios put everything in the training set.
def test_split_data_zero_ratios_keep_everything_in_training():
    calibration, validation, training = split_data(20, 0, 0, SEED)

    assert calibration == []
    assert validation == []
    assert sorted(training) == list(range(20))


@pytest.mark.parametrize("n_images", [0, -1])
def test_split_data_rejects_non_positive_n_images(n_images):
    with pytest.raises(ValueError, match="n_images"):
        split_data(n_images, CAL, VAL, SEED)


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_split_data_rejects_invalid_calibration_ratio(ratio):
    with pytest.raises(ValueError, match="calibration_ratio"):
        split_data(10, ratio, VAL, SEED)


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_split_data_rejects_invalid_validation_ratio(ratio):
    with pytest.raises(ValueError, match="validation_ratio"):
        split_data(10, CAL, ratio, SEED)


# Ratios that leave no training image are rejected.
def test_split_data_rejects_empty_training_set():
    with pytest.raises(ValueError, match="training set"):
        split_data(224, 1.0, VAL, SEED)
    with pytest.raises(ValueError, match="training set"):
        split_data(224, 0.0, 1.0, SEED)
