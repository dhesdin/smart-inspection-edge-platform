import gc
import json
import statistics
from pathlib import Path

import torch
from torch import Tensor

from smart_inspection.config.loader import resolve_config_paths
from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.io.run_id import get_run_id
from smart_inspection.io.save_fig import save_figure
from smart_inspection.io.save_json import save_json
from smart_inspection.models.factory import create_method
from smart_inspection.visualization.plots import generate_plots

# CONSTANTS
# This script generates anomaly figures for cable swap subcategory using specified methods and checkpoints.
CHECKPOINTS = {
    "padim": "padim_cable_cuda_20261003_171414",
    "stfpm": "stfpm_cable_cuda_20261003_171414",
}
CATEGORY = "cable"
SPLIT = "test"
SUBCATEGORY = "cable_swap"
SUBCATEGORY_REPORT = "results_compare_method_bysubcategory_cuda_20261011_162757"


def _collect_scores(method: str, checkpoint: str, models_dir: Path, dataset: AnomalyDataset) -> list[tuple[int, float, Tensor, int]]:
    """
    Collect anomaly scores for a given method and checkpoint on the provided dataset.

    Args:
        method (str): The name of the anomaly detection method.
        checkpoint (str): The checkpoint file name (without extension) to load the model from.
        models_dir (Path): The directory containing the model checkpoints.
        dataset (AnomalyDataset): The dataset to evaluate.

    Returns:
        list[tuple[int, float, Tensor, int]]: A list of tuples containing the index, anomaly score, anomaly map, and label for each sample in dataset.
    """
    if not checkpoint:
        raise ValueError(f"No checkpoint specified for method '{method}'")
    if not (models_dir / f"{checkpoint}.pt").exists():
        raise FileNotFoundError(f"Checkpoint file '{checkpoint}.pt' for method '{method}' not found in '{models_dir}'")
    method_instance = create_method(method_name=method)
    method_instance.load(file_path=models_dir / f"{checkpoint}.pt")

    results = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        score, anomaly_map = method_instance.predict(sample["image"])

        anomaly_map = anomaly_map.cpu()
        results.append((idx, score, anomaly_map, sample["label"]))

    # reset gpu memory cache
    del method_instance
    gc.collect()
    torch.cuda.empty_cache()

    return results


def _score_diagnostic(scores: list[tuple[int, float, Tensor, int]], swap_indices: list[int], dataset: AnomalyDataset) -> dict:
    """
    Compare the scores of the subcategory images with the scores of the normal images.

    Args:
        scores (list[tuple[int, float, Tensor, int]]): The (index, score, anomaly map, label) tuples of one method, in dataset order.
        swap_indices (list[int]): The dataset indices of the images of the subcategory.
        dataset (AnomalyDataset): The dataset the scores come from.

    Returns:
        dict: Normal score median and max, sorted subcategory scores, number of subcategory images above the max normal score,
            global max score and the subcategory it belongs to.
    """
    normal_scores = [r[1] for r in scores if r[3] == 0]
    if not normal_scores:
        raise ValueError("No normal image in the scores, the diagnostic needs a normal reference")

    max_normal = max(normal_scores)
    swap_scores = sorted(scores[i][1] for i in swap_indices)
    best = max(scores, key=lambda r: r[1])

    return {
        "n_normal": len(normal_scores),
        "normal_median": statistics.median(normal_scores),
        "normal_max": max_normal,
        "swap_scores_sorted": swap_scores,
        "n_swap_above_normal_max": sum(s > max_normal for s in swap_scores),
        "global_max": best[1],
        "global_max_subcategory": dataset.samples[best[0]]["subcategory"],
    }


def _argmax_hit(anomaly_map: Tensor, mask: Tensor) -> dict:
    """
    Check whether the hottest pixel of an anomaly map falls inside the ground-truth mask.

    Args:
        anomaly_map (Tensor): Anomaly map of one image, shape (H, W).
        mask (Tensor): Ground-truth mask of the same image, shape (1, H, W).

    Returns:
        dict: Row and column of the max pixel, and whether it is inside the mask.

    Raises:
        ValueError: If the shapes do not match once the mask is squeezed, or if the mask is empty.
    """
    mask_binary = mask.squeeze() > 0.5  # aligne the mask shape with the map
    if mask_binary.shape != anomaly_map.shape:
        raise ValueError(f"Shape mismatch: anomaly_map {anomaly_map.shape}, mask {mask_binary.shape}")
    if not mask_binary.any():
        raise ValueError("Mask is empty")

    flat_index = torch.argmax(input=anomaly_map)  # find idx of the max pixel value
    row, col = divmod(flat_index.item(), anomaly_map.shape[1])  # flat // width, flat % width
    hit = bool(mask_binary[row, col])  # check if the max pixel is inside the mask
    return {"row": row, "col": col, "hit": hit}


def main() -> None:
    """Generate anomaly figures for the cable swap subcategory using specified methods and checkpoints."""

    # config
    common_yaml = resolve_config_paths(config_path="common.yaml")
    models_dir = common_yaml["paths"]["models_dir"]
    figures_dir = common_yaml["paths"]["figures_dir"]
    run_id = get_run_id(tag="cable_swap")
    reports_dir = common_yaml["paths"]["reports_dir"]

    # subcategory report
    subcategory_report = reports_dir / f"{SUBCATEGORY_REPORT}.json"
    with open(subcategory_report) as f:
        subcategory_data = json.load(f)
    by_subcategory = subcategory_data["stfpm"]["by_subcategory"]
    if SUBCATEGORY not in by_subcategory:
        raise ValueError(f"Subcategory '{SUBCATEGORY}' not found in '{SUBCATEGORY_REPORT}.json', available: {sorted(by_subcategory)}")
    n_anomaly = by_subcategory[SUBCATEGORY]["n_anomaly"]

    # dataset
    dataset = AnomalyDataset(category=CATEGORY, split=SPLIT)
    # filter to only include samples from the specified subcategory
    swap_indices = [i for i in range(len(dataset)) if dataset.samples[i]["subcategory"] == SUBCATEGORY]
    if len(swap_indices) != n_anomaly:
        raise ValueError(
            f"Number of swap indices ({len(swap_indices)}) does not match number of samples in subcategory '{SUBCATEGORY}' ({n_anomaly})"
        )

    # scores of every method, in the order of CHECKPOINTS
    all_scores = {
        method: _collect_scores(method=method, checkpoint=checkpoint, models_dir=models_dir, dataset=dataset)
        for method, checkpoint in CHECKPOINTS.items()
    }

    # --- visualization ---
    for method, scores in all_scores.items():
        max_score = max(r[1] for r in scores)
        for idx in swap_indices:
            sample = dataset[idx]
            score = scores[idx][1]
            fig = generate_plots(
                image=sample["image"],
                mask=sample["mask"],
                anomaly_map=scores[idx][2],
                title=f"{method} - #{idx} - score {score:.4g} ({score / max_score:.3f} × max)",
            )
            save_figure(fig=fig, file_path=figures_dir / run_id / f"sample_{idx}_{method}.png")

    # --- diagnostic ---
    diagnostics = {method: _score_diagnostic(scores, swap_indices, dataset) for method, scores in all_scores.items()}
    for method, diagnostic in diagnostics.items():
        print(f"--- {method} ---")
        for k, v in diagnostic.items():
            print(f"{k}: {v}")

    # argmax
    # Here we compute the argmax hit results for each method
    # We iterate over indices of the swap samples and compute the argmax hit for each one
    # add a summary of the number of hits for each method
    argmax_results = {}
    for method, scores in all_scores.items():
        results = []
        for idx in swap_indices:
            mask = dataset[idx]["mask"]
            am = scores[idx][2]
            argmax_result = _argmax_hit(anomaly_map=am, mask=mask)
            results.append({"idx": idx, "argmax_hit": argmax_result})
        n_hit_true = sum(r["argmax_hit"]["hit"] for r in results)
        argmax_results[method] = {"results": results, "n_hit_true": n_hit_true}
        print(f"--- {method} ---")
        print(f"Number of hits: {n_hit_true} / {len(results)}")

    report = {
        "config": {
            "run_id": run_id,
            "category": CATEGORY,
            "split": SPLIT,
            "subcategory": SUBCATEGORY,
            "swap_indices": swap_indices,
            "checkpoints": CHECKPOINTS,
        },
        "diagnostic": diagnostics,
        "argmax_hit": argmax_results,
    }
    save_json(data=report, file_path=reports_dir / f"results_{run_id}.json")


if __name__ == "__main__":
    main()
