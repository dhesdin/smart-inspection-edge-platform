from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch import Tensor
from torch.utils.data import DataLoader


class AnomalyMethod(ABC):
    """
    Abstract base class for anomaly detection Methods
    """

    @property
    @abstractmethod
    def requires_validation(self) -> bool:
        """
        Return whether the method requires a validation set.
        Returns:
            bool: True if a validation set is required, False otherwise.
        """
        pass

    @abstractmethod
    def save(self, file_path: Path) -> None:
        """
        Save the model to the specified file path.
        Args:
            file_path (Path): The path to save the model.
        """
        pass

    @abstractmethod
    def load(self, file_path: Path) -> None:
        """
        Load the model from the specified file path.
        Args:
            file_path (Path): The path to load the model from.
        """
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """
        Return the state needed to restore the trained method.
        The method only exposes its own learned state: it knows nothing about the
        decision threshold, files on disk, or protocol metadata.
        Returns:
            dict[str, Any]: The learned state, with tensors detached and copied to CPU.
        """
        pass

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """
        Restore the trained method from a state returned by get_state().
        The method only restores its own learned state: it knows nothing about the
        decision threshold, files on disk, or protocol metadata.
        Args:
            state (dict[str, Any]): The state returned by get_state().
        Raises:
            ValueError: If a required key is missing from state.
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
