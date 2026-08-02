from torch import Tensor
from torch.utils.data import DataLoader

from smart_inspection.models.base import AnomalyMethod


class PaDiM(AnomalyMethod):
    def __init__(self):
        super().__init__()
        pass

    def fit(self, train_loader: DataLoader) -> None:
        """
        Fit the model using the provided training data loader.
        Args:
            train_loader (DataLoader): The training data loader.
        """
        pass

    def predict(self, image: Tensor) -> tuple[float, Tensor]:

        pass
