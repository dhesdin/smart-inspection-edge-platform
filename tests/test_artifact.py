from pathlib import Path

import numpy as np
import pytest
import torch

from smart_inspection.io import artifact as artifact_module
from smart_inspection.io.artifact import FORMAT_VERSION, build_artifact, get_backbone, load_artifact, save_artifact
from tests.test_evaluate import _FakeMethod

_MODEL_CONFIG = {"backbone": "resnet18", "input_size": 8}


class _StatefulFakeMethod(_FakeMethod):
    """Fake method with a tensor state, to check that get_state() / load_state() survive the round trip."""

    def __init__(self, weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.weights = weights

    def get_state(self) -> dict:
        return {"weights": self.weights.clone()}

    def load_state(self, state: dict) -> None:
        self.weights = state["weights"]


def _build(threshold=0.5, model_state=None) -> dict:
    return build_artifact(
        model_name="fake",
        category="cable",
        model_state=model_state if model_state is not None else {"weights": torch.arange(6.0).reshape(2, 3)},
        threshold=threshold,
        model_config=dict(_MODEL_CONFIG),
        preprocessing={"resize": [8, 8], "normalize_mean": [0.5] * 3, "normalize_std": [0.2] * 3},
        backbone={"weights": "ResNet18_Weights.IMAGENET1K_V1", "weights_url": "https://example.invalid/w.pth"},
        run_id="run_test",
        threshold_percentile=95,
        calibration_count=10,
        threshold_method="linear",
        dataset="mvtec_anomaly_detection",
        seed=42,
    )


@pytest.fixture
def fake_factory(monkeypatch):
    """Route load_artifact() to the fake method and a fixed configuration (no yaml, no weights download)."""
    monkeypatch.setattr(artifact_module, "create_method", lambda method_name: _StatefulFakeMethod())
    monkeypatch.setattr(artifact_module, "load_model_config", lambda model_name: dict(_MODEL_CONFIG))


def test_round_trip_keeps_threshold_and_state(tmp_path, fake_factory):
    original = _StatefulFakeMethod(weights=torch.randn(2, 3))
    artifact = _build(threshold=0.123456789, model_state=original.get_state())
    path = tmp_path / "artifact.pt"

    save_artifact(artifact, path)
    loaded = load_artifact(path, device="cpu")

    assert loaded.threshold == 0.123456789
    assert torch.equal(loaded.method.weights, original.weights)
    assert loaded.category == "cable"
    assert loaded.preprocessing == artifact["preprocessing"]
    assert loaded.traceability == artifact["traceability"]
    assert type(loaded.traceability["git_dirty"]) is bool


def test_load_missing_key_raises(tmp_path, fake_factory):
    artifact = _build()
    del artifact["backbone"]
    path = tmp_path / "artifact.pt"
    torch.save(artifact, path)  # bypass save_artifact(), which validates too

    with pytest.raises(ValueError, match="backbone"):
        load_artifact(path, device="cpu")


def test_load_other_format_version_raises(tmp_path, fake_factory):
    artifact = _build()
    artifact["format_version"] = FORMAT_VERSION + 1
    path = tmp_path / "artifact.pt"
    torch.save(artifact, path)

    with pytest.raises(ValueError, match="format_version"):
        load_artifact(path, device="cpu")


def test_load_config_mismatch_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact_module, "create_method", lambda method_name: _StatefulFakeMethod())
    monkeypatch.setattr(artifact_module, "load_model_config", lambda model_name: {**_MODEL_CONFIG, "input_size": 16})
    path = tmp_path / "artifact.pt"
    save_artifact(_build(), path)

    with pytest.raises(ValueError, match="input_size"):
        load_artifact(path, device="cpu")


@pytest.mark.parametrize("threshold", [np.float64(0.5), float("nan"), float("inf"), 1])
def test_build_rejects_non_finite_or_non_python_float_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        _build(threshold=threshold)


def _failing_save(obj, f) -> None:
    """Stand-in for torch.save that leaves a partial file behind, then fails."""
    Path(f).write_bytes(b"partial")
    raise RuntimeError("serialization failed")


def test_save_is_atomic_when_serialization_fails(tmp_path, monkeypatch):
    path = tmp_path / "artifact.pt"
    monkeypatch.setattr(artifact_module.torch, "save", _failing_save)

    with pytest.raises(RuntimeError, match="serialization failed"):
        save_artifact(_build(), path)

    assert list(tmp_path.iterdir()) == []  # neither the final file nor a temporary file


def test_failed_save_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "artifact.pt"
    save_artifact(_build(threshold=0.25), path)
    monkeypatch.setattr(artifact_module.torch, "save", _failing_save)

    with pytest.raises(RuntimeError, match="serialization failed"):
        save_artifact(_build(threshold=0.75), path)

    monkeypatch.undo()
    assert torch.load(path, weights_only=True)["threshold"] == 0.25
    assert list(tmp_path.iterdir()) == [path]


def test_get_backbone_resolves_configured_weights():
    backbone = get_backbone("resnet18", "IMAGENET1K_V1")

    assert backbone["weights"] == "ResNet18_Weights.IMAGENET1K_V1"
    assert backbone["weights_url"].endswith(".pth")


def test_get_backbone_invalid_weights_raises():
    with pytest.raises(ValueError, match="IMAGENET1K_V1"):  # the message lists the available weights
        get_backbone("resnet18", "NOT_A_WEIGHTS_NAME")
