from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from smart_inspection.models.base import AnomalyMethod


def train(
    method: AnomalyMethod,
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    save_path: Path | None = None,
) -> None:
    """Train method on dataset.

    The datasets are used as given: the split is done by the caller.
    A validation loader is built only if val_dataset is not None.

    Args:
        method (AnomalyMethod): The anomaly detection method to be trained.
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset | None): The validation dataset, if available.
        batch_size (int): Batch size for the dataloaders (required).
        seed (int): Seed for the shuffling of the training loader.
        shuffle (bool, optional): Whether to shuffle the training loader. Defaults to True.
        num_workers (int, optional): Number of workers for the dataloaders. Defaults to 0.
        drop_last (bool, optional): Drop the last incomplete batch of the *training*
            loader only; the validation loader always keeps every sample. Defaults to False.
        save_path (Path | None, optional): The path to save the trained model. Defaults to None.

    Raises:
        ValueError: If the training loader is empty.
        ValueError: If val_dataset is required but not provided.

    Returns:
        None
    """
    generator = torch.Generator().manual_seed(seed)

    requires_validation = method.requires_validation

    if not requires_validation:
        val_dataset = None

    if requires_validation and val_dataset is None:
        raise ValueError("val_dataset is required when the method requires validation.")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        shuffle=shuffle,
        generator=generator,
    )
    if len(train_loader) == 0:
        raise ValueError(f"Empty training loader {len(train_dataset)}, batch_size={batch_size}, drop_last={drop_last}")

    if val_dataset is not None:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False,
        )
    else:
        val_loader = None

    method.fit(train_loader=train_loader, val_loader=val_loader)
    if save_path is not None:
        method.save(save_path)
