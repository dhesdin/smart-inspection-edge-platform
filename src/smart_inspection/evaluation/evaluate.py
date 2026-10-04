import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from smart_inspection.models.base import AnomalyMethod


def evaluate(method: AnomalyMethod, test_loader: DataLoader) -> dict[str, float | int]:
    """Evaluate the given anomaly detection model on the test dataset.
    Args:
        method (AnomalyMethod): The anomaly detection method to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        dict[str, float | int]: A dictionary containing evaluation metrics.
    """
    scores = []
    labels = []
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

    return {
        "roc_auc": float(roc_auc),
        "n_normal": labels.count(0),
        "n_anomaly": labels.count(1),
        "roc_auc_pixel": float(roc_auc_pixel),
        "n_pixel_normal": int((all_pixels_labels == 0).sum()),
        "n_pixel_anomaly": int((all_pixels_labels == 1).sum()),
        "n_samples": len(labels),
    }
