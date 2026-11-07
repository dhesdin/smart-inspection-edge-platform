import numpy as np


def compute_threshold(scores: list, percentile: float) -> float:
    """
    Compute the threshold value for the given scores based on the specified percentile.
    The scores came from the calibration set used to determine the threshold.
    Args:
        scores (list): List of numerical scores.
        percentile (float): Percentile value (0-100) to compute the threshold.
    Raises:
        ValueError: If the scores list is empty, contains NaN, or the percentile is out of range.

    Returns:
        float: The computed threshold value.
    """
    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        raise ValueError("Scores cannot be empty")

    if np.isnan(scores).any():
        raise ValueError("Scores cannot contain NaN")

    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")

    threshold = np.percentile(scores, percentile, method="linear")

    return float(threshold)


def compute_metrics(scores: list, labels: list, threshold: float) -> dict:
    """
    Evaluate the given scores against the provided labels using the specified threshold.
    Args:
        scores (list): List of numerical scores.
        labels (list): List of binary labels (0 or 1).
        threshold (float): Threshold value to classify scores as positive.
    Raises:
        ValueError: If the scores list is empty, contains NaN, labels are not binary,
            lengths mismatch, the threshold is None/NaN, or there are no anomalies
            (positive labels) in the labels.

    Returns:
        dict: Dictionary containing precision, recall, f1 score, and confusion matrix
            elements (TP, FP, FN, TN). Convention: precision = 0.0 if nothing is
            predicted as anomalous (TP + FP == 0).
    """

    # convert into np arrays for compute
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)

    # guard clauses
    if len(scores) == 0:
        raise ValueError("Scores cannot be empty")

    if len(scores) != len(labels):
        raise ValueError("Scores and labels must have the same length")

    if np.isnan(scores).any():
        raise ValueError("Scores cannot contain NaN")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1")

    labels = labels.astype(int)

    if threshold is None or np.isnan(threshold):
        raise ValueError("Threshold cannot be None or NaN")

    # Recall : no anomalies in the labels = invalid input
    if int(np.sum(labels == 1)) == 0:
        raise ValueError("Cannot calculate Recall: no anomalies in labels")

    predictions = (scores >= threshold).astype(int)  # convert boolean array to int (0 or 1)

    # metrics int for serialize into json
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))

    # Precision : no anomalies predicted = 0.0
    precision = float(tp / (tp + fp) if (tp + fp) > 0 else 0.0)

    # recall : real anomalies
    recall = float(tp / (tp + fn))

    # harmonic mean of precision and recall
    f1 = float(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }
