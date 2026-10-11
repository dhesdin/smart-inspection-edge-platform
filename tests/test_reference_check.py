import pytest

from smart_inspection.evaluation.reference_check import compare_with_reference

# Cable counts: 58 normal, 92 anomalous.
# One image AUROC step = 1 / (58 * 92) = 1 / 5336 ~= 1.874062968515742e-4
# Tolerance = 1/10 of a step = (1/10)* STEP ~= 1.874062968515742e-5
STEP = 1.874062968515742e-4
TOLERANCE = 1.874062968515742e-5
REFERENCE_AUROC = 0.9


def _result(roc_auc: float = REFERENCE_AUROC, roc_auc_pixel: float = 0.95, n_normal: int = 58, n_anomaly: int = 92) -> dict:
    """Minimal evaluate() output for one method, with the keys used by compare_with_reference."""
    return {"roc_auc": roc_auc, "roc_auc_pixel": roc_auc_pixel, "n_normal": n_normal, "n_anomaly": n_anomaly}


def _compare(evaluate_result: dict, device: str = "cuda", reference_device: str = "cuda") -> dict:
    return compare_with_reference(evaluate_result, _result(), device, reference_device)


# The tolerance is 1/10 of one AUROC step, computed from the reference counts: 1 / (10 * 58 * 92) = 1 / 53360.
def test_tolerance_is_a_tenth_of_one_auroc_step():
    check = _compare(_result())

    assert check["tolerance_image_auroc"] == pytest.approx(TOLERANCE)


# Identical AUROCs give a zero gap, within tolerance.
def test_identical_auroc_gives_zero_gap_within_tolerance():
    check = _compare(_result())

    assert check["gap_image_auroc"] == pytest.approx(0.0)
    assert check["image_within_tolerance"] is True


# A gap of one full AUROC step (one swapped pair) is 10 times the tolerance: rejected.
def test_gap_of_one_auroc_step_is_not_within_tolerance():
    check = _compare(_result(roc_auc=REFERENCE_AUROC + STEP))

    assert check["gap_image_auroc"] == pytest.approx(STEP)
    assert check["image_within_tolerance"] is False


# Half the tolerance (~9.37e-6) is accepted. Values away from the boundary avoid float rounding on equality.
def test_gap_of_half_tolerance_is_within_tolerance():
    check = _compare(_result(roc_auc=REFERENCE_AUROC + 9.37031484257871e-6))

    assert check["image_within_tolerance"] is True


# Twice the tolerance (~3.75e-5) is rejected.
def test_gap_of_twice_tolerance_is_not_within_tolerance():
    check = _compare(_result(roc_auc=REFERENCE_AUROC + 3.748125937031484e-5))

    assert check["image_within_tolerance"] is False


# The gap is absolute: an evaluation above or below the reference, beyond tolerance, is rejected in both cases.
@pytest.mark.parametrize("roc_auc", [REFERENCE_AUROC + 3.748125937031484e-5, REFERENCE_AUROC - 3.748125937031484e-5])
def test_gap_beyond_tolerance_is_rejected_in_both_directions(roc_auc):
    check = _compare(_result(roc_auc=roc_auc))

    assert check["gap_image_auroc"] == pytest.approx(3.748125937031484e-5)
    assert check["image_within_tolerance"] is False


# One anomalous image missing (58/91 instead of 58/92): counts do not match.
def test_counts_mismatch_is_reported():
    check = _compare(_result(n_anomaly=91))

    assert check["counts_match"] is False


# Same counts as the reference: counts match.
def test_counts_match_is_reported():
    check = _compare(_result())

    assert check["counts_match"] is True


# device_match reports whether the run used the same device as the reference.
@pytest.mark.parametrize(("device", "expected"), [("cuda", True), ("cpu", False)])
def test_device_match(device, expected):
    check = _compare(_result(), device=device, reference_device="cuda")

    assert check["device_match"] is expected


# The pixel gap is measured only: a huge pixel gap (0.95 - 0.45 = 0.5) does not affect the image verdict.
def test_pixel_gap_is_measured_but_does_not_affect_image_verdict():
    check = _compare(_result(roc_auc_pixel=0.45))

    assert check["gap_pixel_auroc"] == pytest.approx(0.5)
    assert check["image_within_tolerance"] is True


# A reference count of 0 would lead to a ValueError.
def test_zero_reference_count_raises_value_error():
    with pytest.raises(ValueError, match="positive"):
        compare_with_reference(_result(), _result(n_normal=0), "cuda", "cuda")


# Same guard on the anomalous count.
def test_zero_reference_anomaly_count_raises_value_error():
    with pytest.raises(ValueError, match="positive"):
        compare_with_reference(_result(), _result(n_anomaly=0), "cuda", "cuda")


# The tolerance follows the reference counts, not fixed cable values. Bottle: 20 * 63 = 1260, tolerance = 1 / 12600.
def test_tolerance_follows_reference_counts():
    bottle = _result(n_normal=20, n_anomaly=63)

    check = compare_with_reference(bottle, bottle, "cuda", "cuda")

    assert check["tolerance_image_auroc"] == pytest.approx(7.936507936507937e-5)


# When the evaluated counts differ (58/91), the tolerance still comes from the reference (58/92): 1 / 53360,
# not 1 / (10 * 58 * 91) = 1 / 52780 ~= 1.8947e-5.
def test_tolerance_uses_reference_counts_when_counts_differ():
    check = _compare(_result(n_anomaly=91))

    assert check["counts_match"] is False
    assert check["tolerance_image_auroc"] == pytest.approx(TOLERANCE)


# The returned keys are exactly the six documented ones, nothing missing and nothing extra.
def test_returned_keys_are_exactly_the_documented_ones():
    check = _compare(_result())

    assert set(check) == {
        "tolerance_image_auroc",
        "gap_image_auroc",
        "gap_pixel_auroc",
        "counts_match",
        "device_match",
        "image_within_tolerance",
    }
