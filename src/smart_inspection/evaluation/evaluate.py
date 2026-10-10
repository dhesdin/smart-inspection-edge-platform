import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from smart_inspection.models.base import AnomalyMethod


def _evaluate_by_subcategory(scores: list[float], labels: list[int], subcategories: list[str]) -> dict[str, dict[str, float | int]]:
    """Compute one image-level ROC AUC per anomaly subcategory.
    For each defect type, the positives are the anomalous samples of that type and the negatives are
    all the normal samples of the test set (the same normal samples are shared by every defect type).
    Args:
        scores (list[float]): Image-level anomaly scores, one per sample.
        labels (list[int]): Ground-truth labels, 0 for normal and 1 for anomalous.
        subcategories (list[str]): Subcategory of each sample (folder name, "good" for normal images).

    Returns:
        dict[str, dict[str, float | int]]: For each anomalous subcategory (sorted by name), its "roc_auc",
        "n_normal" and "n_anomaly".

    Raises:
        ValueError: If the three lists do not have the same length, or if the test set contains
        no normal sample or no anomalous sample.
    """
    scores_by_subcategory = {}
    normal_scores = []
    # Separate normal samples from anomalous samples.
    for score, label, subcategory in zip(scores, labels, subcategories, strict=True):  # strict -> same length
        if label == 0:
            normal_scores.append(score)
        else:
            scores_by_subcategory.setdefault(subcategory, []).append(score)
    # validate test set
    if not normal_scores:
        raise ValueError("The test set must contain at least one normal sample to compute per-subcategory ROC AUC.")

    if not scores_by_subcategory:
        raise ValueError("The test set must contain at least one anomalous sample to compute ROC AUC")
    results = {}
    for subcat, anomalous_data in sorted(scores_by_subcategory.items()):
        # Compute ROC AUC for the current subcategory.
        y_true = [0] * len(normal_scores) + [1] * len(anomalous_data)  # as many 0 as normal samples, as many 1 as anomalous samples
        y_score = normal_scores + anomalous_data  # as many scores as normal samples followed by scores of anomalous samples
        roc_auc = roc_auc_score(y_true=y_true, y_score=y_score)
        results[subcat] = {
            "roc_auc": float(roc_auc),
            "n_normal": int(len(normal_scores)),
            "n_anomaly": len(anomalous_data),
        }
    return results


def evaluate(method: AnomalyMethod, test_loader: DataLoader) -> dict[str, float | int | dict[str, dict[str, float | int]]]:
    """Evaluate the given anomaly detection model on the test dataset.
    Each batch provides image, label, mask, and subcategory.
    Args:
        method (AnomalyMethod): The anomaly detection method to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        dict[str, float | int | dict[str, dict[str, float | int]]]: Image-level and pixel-level ROC AUC with their
        sample/pixel counts, plus "by_subcategory": for each defect type ("good" excluded), its "roc_auc",
        "n_normal" and "n_anomaly" computed against all the normal samples.
    """
    scores = []
    labels = []
    subcategories = []
    pixel_scores = []
    pixel_labels = []

    for batch in test_loader:
        b = batch["image"].shape[0]  # size of batch
        for i in range(b):
            mask = batch["mask"][i]
            score, anomaly_map = method.predict(batch["image"][i])
            if anomaly_map.numel() != mask.numel():
                raise ValueError(f"Anomaly map {tuple(anomaly_map.shape)} does not match mask {tuple(mask.shape)}.")
            scores.append(score)
            labels.append(batch["label"][i].item())
            subcategories.append(batch["subcategory"][i])
            # tolist -> 32 bytes in python , for optimization we use numpy arrays instead
            pixel_scores.append(anomaly_map.flatten().cpu().numpy())  # cpu bc numpy refused gpu tensor
            pixel_labels.append(mask.flatten().numpy().astype(np.uint8))  # one byte array per pixel

    if not labels:
        raise ValueError("test_loader produced no samples.")

    if len(set(labels)) < 2:
        raise ValueError("The test set must contain at least one normal and one anomalous sample to compute ROC AUC.")
    # concatenate replace .extend() to flatten in one array
    all_pixels_labels = np.concatenate(pixel_labels)
    all_pixels_scores = np.concatenate(pixel_scores)
    if len(np.unique(all_pixels_labels)) < 2:
        raise ValueError("The test masks must contain at least one normal and one anomalous pixel to compute pixel-level ROC AUC.")
    roc_auc = roc_auc_score(y_true=labels, y_score=scores)
    roc_auc_pixel = roc_auc_score(y_true=all_pixels_labels, y_score=all_pixels_scores)

    # adding results by subcategory
    results_by_subcategory = _evaluate_by_subcategory(scores=scores, labels=labels, subcategories=subcategories)

    return {
        "roc_auc": float(roc_auc),
        "n_normal": labels.count(0),
        "n_anomaly": labels.count(1),
        "roc_auc_pixel": float(roc_auc_pixel),
        "n_pixel_normal": int((all_pixels_labels == 0).sum()),
        "n_pixel_anomaly": int((all_pixels_labels == 1).sum()),
        "n_samples": len(labels),
        "by_subcategory": results_by_subcategory,
    }
