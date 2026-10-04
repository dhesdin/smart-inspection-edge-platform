import gc
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from smart_inspection.config.loader import resolve_config_paths
from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.evaluation.benchmark import benchmark
from smart_inspection.io.run_id import get_run_id
from smart_inspection.io.save_json import save_json
from smart_inspection.models.factory import create_method

# CONSTANTS
CHECKPOINTS = {
    "stfpm": "stfpm_cable_cuda_20261003_171414",
    "padim": "padim_cable_cuda_20261003_171414",
}
N_IMAGES = 500
N_WARMUP = 10
N_MEASURE = 60


def _benchmark_one(method: str, checkpoint: str, models_dir: Path, images: list[Tensor], device: torch.device) -> dict:
    """
    Benchmark a single method with the given checkpoint and images.

    Args:
        method (str): The name of the method to benchmark.
        checkpoint (str): The checkpoint file name (without extension) to load the model from.
        models_dir (Path): The directory containing the model checkpoints.
        images (list[Tensor]): A list of images to benchmark on.
        device (torch.device): The device to run the benchmark on.

    Returns:
        dict: The benchmark results.
    """
    if not checkpoint:
        raise ValueError(f"No checkpoint specified for method '{method}'")
    if not (models_dir / f"{checkpoint}.pt").exists():
        raise FileNotFoundError(f"Checkpoint file '{checkpoint}.pt' for method '{method}' not found in '{models_dir}'")

    method_instance = create_method(method_name=method)
    method_instance.load(models_dir / f"{checkpoint}.pt")
    benchmark_result = benchmark(
        method=method_instance,
        images=images,
        n_warmup=N_WARMUP,
        n_measures=N_MEASURE,
        device=device,
    )
    # Delete the method instance to free up memory between calls
    del method_instance
    gc.collect()  # garbage collection for freeing up memory after deleting unused ref
    torch.cuda.empty_cache()  # clear GPU mem cache
    return benchmark_result


def main() -> None:
    """Benchmark the FPS and VRAM usage between models."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark FPS and VRAM")
    device = torch.device("cuda")

    # config
    common_yaml = resolve_config_paths(config_path="common.yaml")
    reports_dir = common_yaml["paths"]["reports_dir"]
    models_dir = common_yaml["paths"]["models_dir"]
    run_id = get_run_id(tag=f"benchmark_{device}")
    category_dataset = "cable"

    # dataset : linspace for selecting N_IMAGES evenly spaced samples from the test set
    dataset = AnomalyDataset(category=category_dataset, split="test")
    indices = np.linspace(0, len(dataset) - 1, num=min(N_IMAGES, len(dataset)), dtype=int)
    images = [dataset[int(i)]["image"] for i in indices]
    print(f"Using {len(images)} images for benchmarking.")

    # BENCHMARK
    benchmark_result_stfpm = _benchmark_one(method="stfpm", checkpoint=CHECKPOINTS["stfpm"], models_dir=models_dir, images=images, device=device)
    print(f"STFPM benchmark result: {benchmark_result_stfpm}")

    print(f"Verifying VRAM usage before loading PaDiM: {torch.cuda.memory_allocated() / 1024**2:.4f} MB")
    benchmark_result_padim = _benchmark_one(method="padim", checkpoint=CHECKPOINTS["padim"], models_dir=models_dir, images=images, device=device)
    print(f"PaDiM benchmark result: {benchmark_result_padim}")

    # report
    config = {
        "run_id": run_id,
        "dataset_category": category_dataset,
        "device": str(device),
        "n_images": len(images),
        "n_warmup": N_WARMUP,
        "n_measures": N_MEASURE,
    }
    report = {
        "config": config,
        "stfpm": benchmark_result_stfpm,
        "padim": benchmark_result_padim,
    }
    save_json(data=report, file_path=reports_dir / f"{run_id}_benchmark.json")


if __name__ == "__main__":
    main()
