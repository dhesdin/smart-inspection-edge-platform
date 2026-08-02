from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import DataLoader


class AnomalyMethod(ABC):
    """
    Abstract base class for anomaly detection Methods
    """

    @abstractmethod
    def fit(self, train_loader: DataLoader) -> None:
        """
        Fit the model using the provided training data loader.
        Args:
            train_loader (DataLoader): The training data loader.
        """
        pass

    @abstractmethod
    def predict(self, image: Tensor) -> tuple[float, Tensor]:
        """
        Predict anomalies using the provided test data loader.
        Args:
            image (Tensor): The input image for anomaly prediction.

        Returns:
            tuple[float, Tensor]: A tuple containing a scalar anomaly score and a 2D anomaly map.
        """
        pass
