from __future__ import annotations

from pathlib import Path

import yaml


## ================ PATHS ================ ##
def get_root_dir() -> Path:
    """Get the root directory of the project"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find the project root: no pyproject.toml found in any directory")


def _get_config_path(config_path: str) -> Path:
    """Get the path to the configuration file
    Only Used by read_yaml and resolve_config_paths functions
    Args:
        config_path (str): The path to the configuration file
    Returns:
        Path: The path to the configuration file
    """
    root_dir = get_root_dir()
    config_path = root_dir / "configs" / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    return config_path


def resolve_config_paths(config_path: str) -> dict:
    """Get the paths from the configuration file
    Args:
        config_path (str): The path to the configuration file
    Returns:
        dict: The paths from the configuration file"""
    root_dir = get_root_dir()

    config = read_yaml(config_path=config_path)
    paths = {}
    for k, v in config.items():
        if k in ("paths", "data"):
            sub_paths = {subkey: root_dir / subvalue for subkey, subvalue in v.items()}
            paths[k] = sub_paths

        else:
            paths[k] = v
    return paths


## ================ YAML CONFIGURATION FILES ================ ##
def read_yaml(config_path: str) -> dict:
    """Read the configuration file in yaml format
    and return a dictionary of the configuration parameters
    Args:
        config_path (str): The path to the configuration file
    Returns:
        dict: The configuration parameters"""
    config_path = _get_config_path(config_path)
    # No FileNotFoundError catch needed :  _get_config_path already checks if the file exists
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError:
        raise yaml.YAMLError(f"Configuration file {config_path} is broken")


def merge_yaml(common_config: dict, model_config: dict) -> dict:
    """Merge the common configuration with the model specific configuration
    Args:
        common_config (dict): The common configuration parameters
        model_config (dict): The model specific configuration parameters
    Returns:
        dict: The merged configuration parameters"""
    merge = {}
    for k, v in common_config.items():
        if k in model_config:
            merge[k] = {**common_config[k], **model_config[k]}
        else:
            merge[k] = v
    merge = {**model_config, **merge}  # add the keys that are only in model_config
    return merge
