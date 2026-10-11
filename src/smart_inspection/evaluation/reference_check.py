# Fraction of one image AUROC step (one swapped (normal, anomaly) pair)
# that is allowed as a gap with the reference
TOLERANCE_FRACTION = 1 / 10


def compare_with_reference(evaluate_result: dict, reference_result: dict, device: str, reference_device: str) -> dict:
    """Compare the evaluation of one method with its reference report, without judging the result.

    One swapped (normal, anomaly) pair moves the image AUROC by 1/(n_normal*n_anomaly). The tolerance is
    TOLERANCE_FRACTION of this step, computed from the reference counts. The pixel gap is only measured.
    Args:
        evaluate_result (dict): Output of evaluate() for one method (roc_auc, roc_auc_pixel, n_normal, n_anomaly).
        reference_result (dict): Section of the reference report for the same method, with the same keys.
        device (str): Device of the current run.
        reference_device (str): Device of the reference run.

    Returns:
        dict: "tolerance_image_auroc", "gap_image_auroc", "gap_pixel_auroc", "counts_match", "device_match"
        and "image_within_tolerance".

    Raises:
        ValueError: If a reference count is lower than or equal to 0.
    """
    n_normal = reference_result["n_normal"]
    n_anomaly = reference_result["n_anomaly"]
    if n_normal <= 0 or n_anomaly <= 0:
        raise ValueError(f"Reference counts must be positive, got n_normal={n_normal} and n_anomaly={n_anomaly}.")

    tolerance_image_auroc = TOLERANCE_FRACTION * (1 / (n_normal * n_anomaly))
    gap_image_auroc = abs(evaluate_result["roc_auc"] - reference_result["roc_auc"])
    gap_pixel_auroc = abs(evaluate_result["roc_auc_pixel"] - reference_result["roc_auc_pixel"])  # measured only
    # Check if the counts of normal and anomaly images match the reference
    counts_match = (evaluate_result["n_normal"], evaluate_result["n_anomaly"]) == (n_normal, n_anomaly)
    return {
        "tolerance_image_auroc": tolerance_image_auroc,
        "gap_image_auroc": gap_image_auroc,
        "gap_pixel_auroc": gap_pixel_auroc,
        "counts_match": counts_match,
        # a different device (cpu vs cuda) gives a different rounding noise, which can explain a gap
        "device_match": device == reference_device,
        "image_within_tolerance": gap_image_auroc <= tolerance_image_auroc,
    }
