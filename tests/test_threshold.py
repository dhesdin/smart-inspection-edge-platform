import pytest

from smart_inspection.evaluation.threshold import compute_metrics, compute_threshold


# P50 of [1, 2, 3, 4]: position = 0.50 * (4-1) = 1.5 -> interpolate values[1]=2 and values[2]=3: 2 + 0.5*(3-2) = 2.5
def test_p50():
    values = [1, 2, 3, 4]
    assert compute_threshold(values, 50) == pytest.approx(2.5)


# P95 of [1..44] and the number of values >= T
def test_p95_and_count():
    values = list(range(1, 45))
    threshold = compute_threshold(values, 95)
    count = sum(1 for v in values if v >= threshold)
    # test = (n-1) * 0.95 --> 43 * 0.95 = 40.85 --> 41 + 0.85(42-41) = 41.85
    assert threshold == pytest.approx(41.85)
    # values >= 41.85 -> 3 values
    assert count == 3


# threshold metrics computation
def test_threshold_four_counts():
    scores = [0.1, 0.6, 0.7, 0.4, 0.9]
    labels = [0, 0, 0, 1, 1]
    threshold = 0.5
    metrics = compute_metrics(scores, labels, threshold)
    assert metrics["TP"] == 1
    assert metrics["FP"] == 2
    assert metrics["FN"] == 1
    assert metrics["TN"] == 1
    # precision = tp / (tp+fp) = 1 / (1+2) = 1/3
    assert metrics["precision"] == pytest.approx(1 / 3)
    # recall = tp / (tp+fn) = 1 / (1+1) = 0.5
    assert metrics["recall"] == pytest.approx(0.5)
    # f1 = 2 * precision * recall / (precision+recall) = 2*(1/3)*0.5 / (1/3+0.5) = 0.4
    assert metrics["f1"] == pytest.approx(0.4)


# Test that the threshold boundary is inclusive (>= threshold counts as positive)
def test_threshold_boundary_is_inclusive():
    scores = [0.5, 0.9]
    labels = [0, 1]
    threshold = 0.5
    metrics = compute_metrics(scores, labels, threshold)
    assert metrics["FP"] == 1
    assert metrics["TN"] == 0
    assert metrics["TP"] == 1


# Precision 0/0 -> 0.0
def test_precision_zero_division():
    scores = [0.0]
    labels = [1]
    threshold = 0.5
    metrics = compute_metrics(scores, labels, threshold)
    assert metrics["FP"] == 0
    assert metrics["TP"] == 0
    # precision = tp / (tp+fp)
    assert metrics["precision"] == 0.0


# Recall 0/0 -> should raise ValueError because there are no anomalies in labels
def test_recall_zero_division():
    scores = [0.0]
    labels = [0]
    threshold = 0.5
    # no anomalie here
    # tp / tp+fn
    with pytest.raises(ValueError, match="Cannot calculate Recall: no anomalies in labels"):
        compute_metrics(scores, labels, threshold)


#  labels 0.5 and 2
def test_invalid_labels():
    with pytest.raises(ValueError, match="Labels must contain only 0 and 1"):
        compute_metrics([0.1, 0.9], [0, 0.5], 0.5)

    with pytest.raises(ValueError, match="Labels must contain only 0 and 1"):
        compute_metrics([0.1, 0.9], [0, 2], 0.5)


# NaN (scores and threshold)
def test_nan_scores():
    with pytest.raises(ValueError, match="Scores cannot contain NaN"):
        compute_metrics([0.1, float("nan")], [0, 1], 0.5)


def test_nan_threshold():
    with pytest.raises(ValueError, match="Threshold cannot be None or NaN"):
        compute_metrics([0.1, 0.9], [0, 1], float("nan"))


def test_none_threshold():
    with pytest.raises(ValueError, match="Threshold cannot be None or NaN"):
        compute_metrics([0.1, 0.9], [0, 1], None)


def test_mismatched_lengths():
    with pytest.raises(ValueError, match="Scores and labels must have the same length"):
        compute_metrics([0.1, 0.9], [0], 0.5)


def test_empty_metrics_inputs():
    with pytest.raises(ValueError, match="Scores cannot be empty"):
        compute_metrics([], [], 0.5)


# compute_threshold guard clauses: empty scores, NaN scores, percentile out of range.
def test_empty_threshold_scores():
    with pytest.raises(ValueError, match="Scores cannot be empty"):
        compute_threshold([], 50)


def test_nan_threshold_scores():
    with pytest.raises(ValueError, match="Scores cannot contain NaN"):
        compute_threshold([0.1, float("nan")], 50)


def test_percentile_below_range():
    with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
        compute_threshold([1, 2, 3], -1)


def test_percentile_above_range():
    with pytest.raises(ValueError, match="Percentile must be between 0 and 100"):
        compute_threshold([1, 2, 3], 101)


# Test output types of compute_metrics and compute_threshold functions
def test_output_types():
    metrics = compute_metrics([0.1, 0.9], [0, 1], 0.5)
    for key in ["TP", "FP", "TN", "FN"]:
        assert type(metrics[key]) is int
    for key in ["precision", "recall", "f1"]:
        assert type(metrics[key]) is float


def test_compute_threshold_output_type():
    assert type(compute_threshold([1, 2, 3, 4], 50)) is float
