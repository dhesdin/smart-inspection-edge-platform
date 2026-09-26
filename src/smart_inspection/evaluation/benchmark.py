import time

import numpy as np
import torch
from torch import Tensor

from smart_inspection.models.base import AnomalyMethod


def benchmark(
    method: AnomalyMethod,
    images: list[Tensor],
    n_warmup: int,
    n_measures: int,
    device: torch.device,
) -> dict[str, float | int]:
    """
    Benchmark the given anomaly detection method on a set of images.

    Args:
        method (AnomalyMethod): The anomaly detection method to benchmark.
        images (list[Tensor]): List of input images as PyTorch tensors.
        n_warmup (int): Number of warmup iterations to stabilize GPU performance.
        n_measures (int): Number of measurement iterations.
        device (torch.device): The CUDA device to use for benchmarking.

    Returns:
        dict[str, float | int]: A dictionary containing benchmark results, including:
            - median_t_transfer_ms: Median transfer time in milliseconds.
            - median_compute_t_ms: Median compute time in milliseconds.
            - median_total_time_ms: Median total time (transfer + compute) in milliseconds.
            - percentile_95_total_time_ms: 95th percentile of total time in milliseconds.
            - percentile_99_total_time_ms: 99th percentile of total time in milliseconds.
            - fps: Frames per second based on median total time.
            - resident_vram_mb: Resident VRAM in megabytes take after warmup.
            - peak_vram_mb: Peak VRAM usage in megabytes.
            - inference_cost_vram_mb: extra cost VRAM used during inference in megabytes.
    """

    # handle errors
    if device.type != "cuda":
        raise ValueError(f"Device must be a CUDA device, but got device type '{device.type}'")
    if not images:
        raise ValueError("images must be not empty")

    if n_warmup < 0:
        raise ValueError("n_warmup should be at least 0")
    if n_measures < 1:
        raise ValueError("n_measures should be at least 1")

    # WARMUP loop to stabilize GPU performance
    for i in range(n_warmup):
        image = images[i % len(images)]  # ensure loop over images
        image = image.to(device)
        method.predict(image)

    # measurement
    torch.cuda.synchronize(device=device)  # ensure all ops of CUDA are finished before timing

    # resident VRAM: memory buffer at rest (model weights,...)
    resident_vram_mb = torch.cuda.memory_allocated(device=device) / (1024**2)

    transfer_times = []
    compute_times = []
    total_times = []
    vram_peak_mb_list = []

    for i in range(n_measures):
        image = images[i % len(images)]

        # reset vram max mem at each iter
        torch.cuda.reset_peak_memory_stats(device=device)

        # -------------
        #   Transfer
        # -------------
        transfer_time_start = time.perf_counter()
        image_on_device = image.to(device)
        torch.cuda.synchronize(device=device)  # ensure all ops of CUDA are finished before timing
        transfer_time_end = time.perf_counter()

        # -------------
        #   Compute
        # -------------
        compute_time_start = time.perf_counter()
        method.predict(image=image_on_device)
        torch.cuda.synchronize(device=device)  # ensure all ops of CUDA are finished before timing
        compute_time_end = time.perf_counter()

        # -------------
        #   VRAM
        # -------------
        vram_peak_bytes = torch.cuda.max_memory_allocated(device=device)
        vram_peak_mb = vram_peak_bytes / (1024**2)

        # append
        transfer_times.append(transfer_time_end - transfer_time_start)
        compute_times.append(compute_time_end - compute_time_start)
        vram_peak_mb_list.append(vram_peak_mb)
        # End to end total = transfer + compute
        total_times.append((transfer_time_end - transfer_time_start) + (compute_time_end - compute_time_start))

    # statistics
    median_t_transfer_ms = np.median(transfer_times) * 1000
    median_compute_t_ms = np.median(compute_times) * 1000

    total_times_ms = [t * 1000 for t in total_times]  # in ms
    median_total_time_ms = np.median(total_times_ms)
    percentile_95_total_time_ms = np.percentile(total_times_ms, 95)
    percentile_99_total_time_ms = np.percentile(total_times_ms, 99)
    peak_vram_mb = max(vram_peak_mb_list)
    inference_cost_vram_mb = peak_vram_mb - resident_vram_mb
    fps = 1000 / median_total_time_ms

    # output
    return {
        "median_t_transfer_ms": median_t_transfer_ms,
        "median_compute_t_ms": median_compute_t_ms,
        "median_total_time_ms": median_total_time_ms,
        "percentile_95_total_time_ms": percentile_95_total_time_ms,
        "percentile_99_total_time_ms": percentile_99_total_time_ms,
        "fps": fps,
        "resident_vram_mb": resident_vram_mb,
        "peak_vram_mb": peak_vram_mb,
        "inference_cost_vram_mb": inference_cost_vram_mb,
    }
