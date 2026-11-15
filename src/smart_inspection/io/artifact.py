"""Self-contained deployment artifact: trained method state, decision threshold and traceability."""

import datetime
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision

from smart_inspection.config.loader import get_root_dir, merge_yaml, read_yaml, resolve_config_paths
from smart_inspection.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from smart_inspection.models.base import AnomalyMethod
from smart_inspection.models.factory import create_method
from smart_inspection.models.weights import resolve_backbone_weights

FORMAT_VERSION = 1

# top-level keys of an artifact and their expected type
REQUIRED_KEYS: dict[str, type] = {
    "format_version": int,
    "model_name": str,
    "category": str,
    "model_state": dict,
    "threshold": float,
    "model_config": dict,
    "preprocessing": dict,
    "backbone": dict,
    "traceability": dict,
}

TRACEABILITY_KEYS: tuple[str, ...] = (
    "run_id",
    "threshold_percentile",
    "calibration_count",
    "threshold_method",
    "dataset",
    "seed",
    "git_commit",
    "git_dirty",
    "torch_version",
    "torchvision_version",
    "cuda_version",
    "device_name",
    "created_at",
)


@dataclass
class LoadedArtifact:
    """A restored artifact: the method is ready for predict(), the threshold applies as `score >= threshold`."""

    method: AnomalyMethod
    threshold: float
    category: str
    preprocessing: dict[str, Any]
    traceability: dict[str, Any]


def load_model_config(model_name: str) -> dict[str, Any]:
    """
    Return the merged "params" section (common.yaml + <model_name>.yaml), as read by the method constructors.
    Args:
        model_name (str): The method name, e.g. "padim" or "stfpm".
    Returns:
        dict[str, Any]: The merged parameters.
    """
    common_config = resolve_config_paths(config_path="common.yaml")
    model_config = read_yaml(config_path=f"{model_name}.yaml")
    return merge_yaml(common_config=common_config, model_config=model_config)["params"]


def get_preprocessing() -> dict[str, Any]:
    """
    Return the image preprocessing applied by AnomalyDataset (Resize -> ToTensor -> Normalize).
    The input size comes from common.yaml and the normalization from the dataset module constants.
    The Resize interpolation is not set explicitly by AnomalyDataset (torchvision default), so it is not recorded.
    Returns:
        dict[str, Any]: {"resize", "normalize_mean", "normalize_std"}.
    """
    input_size = resolve_config_paths(config_path="common.yaml")["params"]["input_size"]
    return {
        "resize": [input_size, input_size],
        "normalize_mean": list(IMAGENET_MEAN),
        "normalize_std": list(IMAGENET_STD),
    }


def get_backbone(backbone_name: str, backbone_weights: str) -> dict[str, Any]:
    """
    Return the identifier of the pretrained weights, resolved exactly as the methods resolve them.
    The names themselves are not repeated here: they are model_config["backbone"] and model_config["backbone_weights"].
    Args:
        backbone_name (str): The torchvision model name, e.g. "resnet18".
        backbone_weights (str): The weights name, e.g. "IMAGENET1K_V1".
    Returns:
        dict[str, Any]: {"weights", "weights_url"}.
    Raises:
        ValueError: If backbone_weights is not a weights name of this backbone.
    """
    weights = resolve_backbone_weights(backbone=backbone_name, backbone_weights=backbone_weights)
    return {"weights": str(weights), "weights_url": weights.url}


def _git_output(*args: str) -> str:
    """Run a git command at the project root and return its stdout."""
    result = subprocess.run(["git", *args], cwd=get_root_dir(), capture_output=True, text=True, check=True)
    return result.stdout


def build_artifact(
    *,
    model_name: str,
    category: str,
    model_state: dict[str, Any],
    threshold: float,
    model_config: dict[str, Any],
    preprocessing: dict[str, Any],
    backbone: dict[str, Any],
    run_id: str,
    threshold_percentile: float,
    calibration_count: int,
    threshold_method: str,
    dataset: str,
    seed: int,
) -> dict[str, Any]:
    """
    Assemble an artifact; git and runtime environment information is collected here.
    Args:
        model_name (str): The method name, as accepted by create_method().
        category (str): The dataset category the method was trained on.
        model_state (dict[str, Any]): The output of method.get_state().
        threshold (float): The decision threshold, a finite Python float.
        model_config (dict[str, Any]): The merged method parameters (see load_model_config()).
        preprocessing (dict[str, Any]): The input preprocessing (see get_preprocessing()).
        backbone (dict[str, Any]): The pretrained weights identifier (see get_backbone()).
        run_id (str): The run identifier.
        threshold_percentile (float): The percentile of the calibration scores used for the threshold.
        calibration_count (int): The number of calibration images.
        threshold_method (str): The percentile interpolation method.
        dataset (str): The dataset name.
        seed (int): The seed of the run.
    Returns:
        dict[str, Any]: The artifact.
    Raises:
        ValueError: If threshold is not a finite Python float (numpy floats are rejected).
    """
    if type(threshold) is not float or not math.isfinite(threshold):
        raise ValueError(f"threshold must be a finite Python float, got {threshold!r} ({type(threshold).__name__})")

    cuda_available = torch.cuda.is_available()
    traceability = {
        "run_id": run_id,
        "threshold_percentile": threshold_percentile,
        "calibration_count": calibration_count,
        "threshold_method": threshold_method,
        "dataset": dataset,
        "seed": seed,
        "git_commit": _git_output("rev-parse", "HEAD").strip(),
        "git_dirty": _git_output("status", "--porcelain").strip() != "",
        # str(): torch.__version__ is a TorchVersion, which torch.load(weights_only=True) refuses
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    return {
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "category": category,
        "model_state": model_state,
        "threshold": threshold,
        "model_config": model_config,
        "preprocessing": preprocessing,
        "backbone": backbone,
        "traceability": traceability,
    }


def _validate_artifact(artifact: Any) -> None:
    """
    Check the format version, the required keys and their types, and the threshold.
    Raises:
        ValueError: On the first problem found.
    """
    if not isinstance(artifact, dict):
        raise ValueError(f"artifact must be a dict, got {type(artifact).__name__}")
    if artifact.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported artifact format_version {artifact.get('format_version')!r}, expected {FORMAT_VERSION}")
    missing_keys = [key for key in REQUIRED_KEYS if key not in artifact]
    if missing_keys:
        raise ValueError(f"Artifact is missing required keys: {missing_keys}")
    for key, expected_type in REQUIRED_KEYS.items():
        if not isinstance(artifact[key], expected_type):
            raise ValueError(f"Artifact key {key!r} must be {expected_type.__name__}, got {type(artifact[key]).__name__}")
    if not math.isfinite(artifact["threshold"]):
        raise ValueError(f"Artifact threshold must be finite, got {artifact['threshold']!r}")
    missing_traceability = [key for key in TRACEABILITY_KEYS if key not in artifact["traceability"]]
    if missing_traceability:
        raise ValueError(f"Artifact traceability is missing required keys: {missing_traceability}")


def save_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    """
    Validate and write the artifact atomically: a temporary file in the same directory, then os.replace().
    If serialization fails, no file is left at `path` (nor any temporary file).
    Args:
        artifact (dict[str, Any]): The artifact returned by build_artifact().
        path (str | Path): The destination file.
    Raises:
        ValueError: If the artifact is invalid.
    """
    _validate_artifact(artifact)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(artifact, tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_artifact(path: str | Path, device: str | torch.device) -> LoadedArtifact:
    """
    Load an artifact and restore its method.
    create_method() takes no configuration: the method is built from the current yaml files, so loading is
    refused if they no longer match the artifact's model_config.
    Args:
        path (str | Path): The artifact file.
        device (str | torch.device): The map_location for torch.load(); load_state() then moves the state
            to the method's own device.
    Returns:
        LoadedArtifact: The restored method, threshold, category, preprocessing and traceability.
    Raises:
        ValueError: If the artifact is invalid, has another format_version, or its model_config differs
            from the current configuration.
    """
    artifact = torch.load(path, map_location=device, weights_only=True)
    _validate_artifact(artifact)

    model_name = artifact["model_name"]
    saved_config = artifact["model_config"]
    current_config = load_model_config(model_name)
    if current_config != saved_config:
        differing_keys = sorted(key for key in saved_config.keys() | current_config.keys() if saved_config.get(key) != current_config.get(key))
        raise ValueError(f"Current {model_name} configuration differs from the artifact on keys: {differing_keys}")

    method = create_method(method_name=model_name)
    method.load_state(artifact["model_state"])
    return LoadedArtifact(
        method=method,
        threshold=artifact["threshold"],
        category=artifact["category"],
        preprocessing=artifact["preprocessing"],
        traceability=artifact["traceability"],
    )
