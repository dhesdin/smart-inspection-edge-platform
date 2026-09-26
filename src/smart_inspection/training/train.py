from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.models.base import AnomalyMethod


def train(
    method: AnomalyMethod,
    dataset: AnomalyDataset,
    batch_size: int,
    seed: int | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    save_path: Path | None = None,
) -> None:
    """Train method on dataset.

    The dataset is split into train/validation only when the method requires it
    (method.validation_split_ratio is not None); otherwise the whole dataset
    is used for training and no validation loader is built.

    Args:
        method (AnomalyMethod): The anomaly detection method to be trained.
        dataset (AnomalyDataset): The dataset containing training samples.
        batch_size (int): Batch size for the dataloaders (required).
        seed (int | None, optional): Seed for the train/validation split and for
            the shuffling of the training loader. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the training loader. Defaults to True.
        num_workers (int, optional): Number of workers for the dataloaders. Defaults to 0.
        drop_last (bool, optional): Drop the last incomplete batch of the *training*
            loader only; the validation loader always keeps every sample. Defaults to False.
        save_path (Path | None, optional): The path to save the trained model. Defaults to None.

    Raises:
        ValueError: If validation_split_ratio is not in (0, 1), if the split
            produces an empty subset, or if the resulting training loader is empty.

    Returns:
        None
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    validation_split_ratio = method.validation_split_ratio
    if validation_split_ratio is None:
        train_dataset = dataset
        val_dataset = None
    else:
        if not 0.0 < validation_split_ratio < 1.0:
            raise ValueError(f"validation_split_ratio must be in (0, 1), got {validation_split_ratio!r}")

        n_total = len(dataset)
        n_val = int(n_total * validation_split_ratio)
        n_train = n_total - n_val
        if n_train == 0 or n_val == 0:
            raise ValueError(
                f"Train/validation split produced an empty subset "
                f"(n_total={n_total}, n_train={n_train}, n_val={n_val}) "
                f"validation_split_ratio={validation_split_ratio}"
            )

        train_dataset, val_dataset = random_split(
            dataset,
            [n_train, n_val],
            generator=generator,
        )

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
