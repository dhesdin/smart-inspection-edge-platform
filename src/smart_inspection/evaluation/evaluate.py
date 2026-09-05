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

    for batch in test_loader:
        b = batch["image"].shape[0]  # size of batch
        for i in range(b):
            score, _ = method.predict(batch["image"][i])
            scores.append(score)
            labels.append(batch["label"][i].item())

    if not labels:
        raise ValueError("test_loader produced no samples.")

    if len(set(labels)) < 2:
        raise ValueError("The test set must contain at least one normal and one anomalous sample to compute ROC AUC.")

    roc_auc = roc_auc_score(y_true=labels, y_score=scores)

    return {
        "roc_auc": float(roc_auc),
        "n_samples": len(labels),
        "n_normal": labels.count(0),
        "n_anomaly": labels.count(1),
    }
