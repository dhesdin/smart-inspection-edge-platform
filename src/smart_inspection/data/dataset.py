from pathlib import Path

from smart_inspection.config.loader import resolve_config_paths


class AnomalyDataset:
    def __init__(self, category: str, split: str):
        """
        Constructor for the AnomalyDataset class.
        Args:
            category (str): The category of the dataset (e.g., "bottle", "cable", etc.)
            split (str): The split of the dataset (e.g., "train", "test")
        """
        self.category = category
        self.split = split

        # get paths
        config = resolve_config_paths(config_path="common.yaml")
        dataset_path = config["data"]["datasets"]
        category_path = dataset_path / category
        split_path = category_path / split

        if not split_path.exists():
            raise FileNotFoundError(f"Split path not found at {split_path} it should be called 'train' or 'test'")

        self.samples = self._discover_samples(split_path)  # Pass the split_path to the method

    def _discover_samples(self, split_path: Path) -> list:
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
                    samples.append({"images_path": image, "label": label, "mask_path": split_path.parent / "ground_truth" / p.name / mask_filename})
                else:
                    samples.append({"images_path": image, "label": label, "mask_path": None})
        return samples

    def __len__(self) -> int:
        pass  # TODO : Implement the logic

    def __getitem__(self, idx: int) -> dict:
        pass  # TODO: Implement the logic
