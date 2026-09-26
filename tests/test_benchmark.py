from pathlib import Path

import pytest
import torch
from torch import Tensor

from smart_inspection.evaluation.benchmark import benchmark
from smart_inspection.models.base import AnomalyMethod

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

EXPECTED_KEYS = {
    "median_t_transfer_ms",
    "median_compute_t_ms",
    "median_total_time_ms",
    "percentile_95_total_time_ms",
    "percentile_99_total_time_ms",
    "fps",
    "resident_vram_mb",
    "peak_vram_mb",
    "inference_cost_vram_mb",
}


class _FakeMethod(AnomalyMethod):
    """Minimal AnomalyMethod that records what predict receives and allocates a known amount of VRAM."""

    def __init__(self) -> None:
        self.seen_values: list[float] = []
        self.seen_devices: list[str] = []

    @property
    def validation_split_ratio(self) -> float | None:
        return None

    def save(self, file_path: Path) -> None:
        pass

    def load(self, file_path: Path) -> None:
        pass

    def fit(self, train_loader, val_loader=None) -> None:
        pass

    def predict(self, image: Tensor) -> tuple[float, Tensor]:
        self.seen_values.append(image[0, 0, 0].item())
        self.seen_devices.append(image.device.type)
        _scratch = torch.empty(1024 * 1024, device=image.device)  # 4 MiB, freed when predict returns
        return 0.0, image[0]


def _images(values: tuple[float, ...] = (0.0,)) -> list[Tensor]:
    """One (3, 8, 8) CPU image per value, filled with that value so predict can tell them apart."""
    return [torch.full((3, 8, 8), value) for value in values]


# ---------------- argument validation (no GPU needed: errors are raised before any CUDA call) ---------------- #


# The benchmark refuses to run on a CPU device.
def test_benchmark_rejects_cpu_device():
    with pytest.raises(ValueError, match="Device must be a CUDA device"):
        benchmark(_FakeMethod(), _images(), n_warmup=1, n_measures=1, device=torch.device("cpu"))


# An empty image list is rejected.
def test_benchmark_rejects_empty_images():
    with pytest.raises(ValueError, match="images must be not empty"):
        benchmark(_FakeMethod(), [], n_warmup=1, n_measures=1, device=torch.device("cuda"))


# A negative number of warmup iterations is rejected.
def test_benchmark_rejects_negative_warmup():
    with pytest.raises(ValueError, match="n_warmup"):
        benchmark(_FakeMethod(), _images(), n_warmup=-1, n_measures=1, device=torch.device("cuda"))


# Zero measurements is rejected (statistics would be computed on an empty list).
def test_benchmark_rejects_zero_measures():
    with pytest.raises(ValueError, match="n_measures"):
        benchmark(_FakeMethod(), _images(), n_warmup=1, n_measures=0, device=torch.device("cuda"))


# The method must never be called when the arguments are invalid.
def test_benchmark_does_not_call_predict_on_invalid_arguments():
    method = _FakeMethod()
    with pytest.raises(ValueError):
        benchmark(method, _images(), n_warmup=1, n_measures=0, device=torch.device("cuda"))
    assert method.seen_values == []


# ---------------- behaviour (needs a CUDA GPU) ---------------- #


# The result exposes exactly the documented keys.
@requires_cuda
def test_benchmark_returns_expected_keys():
    result = benchmark(_FakeMethod(), _images(), n_warmup=1, n_measures=3, device=torch.device("cuda"))
    assert set(result.keys()) == EXPECTED_KEYS


# predict is called n_warmup + n_measures times in total.
@requires_cuda
def test_benchmark_calls_predict_warmup_plus_measures_times():
    method = _FakeMethod()
    benchmark(method, _images(), n_warmup=3, n_measures=4, device=torch.device("cuda"))
    assert len(method.seen_values) == 7


# n_warmup=0 is valid: only the measured calls happen.
@requires_cuda
def test_benchmark_accepts_zero_warmup():
    method = _FakeMethod()
    benchmark(method, _images(), n_warmup=0, n_measures=2, device=torch.device("cuda"))
    assert len(method.seen_values) == 2


# Images are cycled over the list, and the warmup loop restarts from the first image.
@requires_cuda
def test_benchmark_cycles_over_images():
    method = _FakeMethod()
    benchmark(method, _images((0.0, 1.0)), n_warmup=3, n_measures=4, device=torch.device("cuda"))
    assert method.seen_values == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    

# predict always receives an image already moved on the GPU, both during warmup and measurement.
@requires_cuda
def test_benchmark_moves_images_to_device_before_predict():
    method = _FakeMethod()
    benchmark(method, _images(), n_warmup=2, n_measures=2, device=torch.device("cuda"))
    assert set(method.seen_devices) == {"cuda"}


# Timings are positive and ordered: median <= p95 <= p99, and total >= compute and >= transfer.
@requires_cuda
def test_benchmark_timings_are_consistent():
    result = benchmark(_FakeMethod(), _images(), n_warmup=2, n_measures=20, device=torch.device("cuda"))
    assert result["median_t_transfer_ms"] > 0
    assert result["median_compute_t_ms"] > 0
    assert result["median_total_time_ms"] >= result["median_compute_t_ms"]
    assert result["median_total_time_ms"] >= result["median_t_transfer_ms"]
    assert result["median_total_time_ms"] <= result["percentile_95_total_time_ms"]
    assert result["percentile_95_total_time_ms"] <= result["percentile_99_total_time_ms"]


# FPS is derived from the median total latency: fps = 1000 / median_total_time_ms.
@requires_cuda
def test_benchmark_fps_matches_median_total_time():
    result = benchmark(_FakeMethod(), _images(), n_warmup=1, n_measures=5, device=torch.device("cuda"))
    assert result["fps"] == pytest.approx(1000 / result["median_total_time_ms"])


# VRAM: peak >= resident, inference cost = peak - resident, and it covers the 4 MiB allocated by predict.
@requires_cuda
def test_benchmark_vram_values_are_consistent():
    result = benchmark(_FakeMethod(), _images(), n_warmup=1, n_measures=3, device=torch.device("cuda"))
    assert result["peak_vram_mb"] >= result["resident_vram_mb"]
    assert result["inference_cost_vram_mb"] == pytest.approx(result["peak_vram_mb"] - result["resident_vram_mb"])
    assert result["inference_cost_vram_mb"] >= 4.0
