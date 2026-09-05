from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import DataLoader


class AnomalyMethod(ABC):
    """
    Abstract base class for anomaly detection Methods
    """

    @property
    @abstractmethod
    def validation_split_ratio(self) -> float | None:
        """
        Return the ratio of the validation split.
        Returns:
            float | None: The validation split ratio.
        """
        pass

    @abstractmethod
    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        """
        Fit the model using the provided training data loader.
        Args:
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader | None, optional): The validation data loader. Defaults to None.
        """
        pass

    @abstractmethod
    def predict(self, image: Tensor) -> tuple[float, Tensor]:
        """
        Predict anomalies for the given input image.
        Args:
            image (Tensor): The input image for anomaly prediction.

        Returns:
            tuple[float, Tensor]: A tuple containing a scalar anomaly score and a 2D anomaly map.
        """
        pass
