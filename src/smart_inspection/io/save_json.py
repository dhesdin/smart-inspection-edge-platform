import json
from pathlib import Path


def save_json(data: dict, file_path: str | Path) -> None:
    """Save a dictionary as a JSON file.

    Args:
        data (dict): The dictionary to save.
        file_path (str | Path): The path to the JSON file.

    Raises:
        ValueError: If the data dictionary is empty.
    """
    if not data:
        raise ValueError("Data dictionary is empty.")
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.parent.exists():
        print(f"Directory does not exist. Creating directory: {file_path.parent}")
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"JSON data saved to {file_path}")
