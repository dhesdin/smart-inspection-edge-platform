from torch.utils.data import Dataset

from smart_inspection.models.base import AnomalyMethod


def collect_scores(method: AnomalyMethod, dataset: Dataset) -> tuple[list[float], list[int]]:
    """Collect anomaly scores and labels for a given method and dataset.

    Args:
        method (AnomalyMethod): The anomaly detection method to use for prediction.
        dataset (Dataset): The dataset to evaluate.
    Raises:
        ValueError: If the dataset is empty.
    Returns:
        tuple[list[float], list[int]]: A tuple containing a list of anomaly scores and a list of corresponding labels.
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    scores = []
    labels = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        score, _ = method.predict(sample["image"])
        scores.append(float(score))
        labels.append(int(sample["label"]))
    return scores, labels
