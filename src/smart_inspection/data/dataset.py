from smart_inspection.config.loader import resolve_config_paths


class AnomalyDataset:
    def __init__(self, category: str, split: str):
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


        pass  # TODO : Implement the logic

    def _discover_samples(self, split_path) -> list:
        pass  # TODO : Implement the logic

    def __len__(self) -> int:
        pass  # TODO : Implement the logic

    def __getitem__(self, idx: int) -> dict:
        pass  # TODO: Implement the logic
