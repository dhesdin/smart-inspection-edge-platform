from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from smart_inspection.config.loader import resolve_config_paths

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AnomalyDataset:
    """AnomalyDataset class for loading and transforming anomaly detection datasets.
    This class is designed to handle datasets for anomaly detection tasks, specifically for the MVTec Anomaly Detection dataset.
    It supports loading images and their corresponding masks, applying transformations, and providing access to the samples in a lazy loading manner.
    """

    def __init__(self, category: str, split: str):
        """
        Constructor for the AnomalyDataset class.
        Args:
            category (str): The category of the dataset (e.g., "bottle", "cable", etc.)
            split (str): The split of the dataset (e.g., "train", "test")
        Raises:
            FileNotFoundError: If the category or split path does not exist.
            ValueError: If no samples are found in the dataset for the given category and split.
        """
        self.category = category
        self.split = split

        # get paths
        config = resolve_config_paths(config_path="common.yaml")
        config_input_size = config["params"]["input_size"]
        self.input_size = (config_input_size, config_input_size)

        dataset_path = config["data"]["datasets"]
        category_path = dataset_path / category
        split_path = category_path / split

        if not category_path.exists():
            raise FileNotFoundError(f"Category path not found at {category_path}")

        if not split_path.exists():
            raise FileNotFoundError(f"Split path not found at {split_path}")

        self.samples = self._discover_samples(split_path)  # Pass the split_path to the method
        self.transform_image = Compose(
            [
                Resize(self.input_size),
                ToTensor(),  # 0-1
                Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.transform_mask = Compose(
            [
                Resize(self.input_size),
                ToTensor(),  # 0-1
            ]
        )

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in the dataset for category '{category}' and split '{split}'.")

    def _discover_samples(self, split_path: Path) -> list:
        """Discover samples in the dataset split directory.
        Args:
            split_path (Path): The path to the dataset split directory.
        Returns:
            list: A list of dictionaries containing sample information.
        """
        samples = []
        for p in split_path.iterdir():
            if not p.is_dir():
                continue
            if p.name == "good":
                label = 0
            else:
                label = 1
            for image in p.glob("*.png"):
                mask_filename = image.stem + "_mask" + image.suffix
                if label == 1:
                    samples.append(
                        {
                            "images_path": image,
                            "label": label,
                            "mask_path": split_path.parent / "ground_truth" / p.name / mask_filename,
                        }
                    )
                else:
                    samples.append({"images_path": image, "label": label, "mask_path": None})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset. For Lazy loading, the image and mask are loaded and transformed when accessed.
        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
            dict: A dictionary containing the image, label, and mask.
        """
        sample = self.samples[idx]
        # transform image
        with Image.open(sample["images_path"]) as image:
            image = self.transform_image(image)
        # mask
        if sample["mask_path"] is not None:
            with Image.open(sample["mask_path"]) as mask:
                mask = self.transform_mask(mask)
        else:
            mask = torch.zeros(1, *self.input_size)

        return {"image": image, "label": sample["label"], "mask": mask}
