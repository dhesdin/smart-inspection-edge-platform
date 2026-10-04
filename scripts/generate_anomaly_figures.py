import gc
from pathlib import Path

import torch
from torch import Tensor

from smart_inspection.config.loader import resolve_config_paths
from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.io.run_id import get_run_id
from smart_inspection.io.save_fig import save_figure
from smart_inspection.models.factory import create_method
from smart_inspection.visualization.plots import generate_comparison_plot, generate_plots

# CONSTANTS
# the order of the methods gives the order of the anomaly map columns in the comparison figures
CHECKPOINTS = {
    "padim": "padim_cable_cuda_20261003_171414",
    "stfpm": "stfpm_cable_cuda_20261003_171414",
}
CATEGORY = "cable"
SPLIT = "test"
TOP_K = 3


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


def _top_k(
    results: list[tuple[int, float, Tensor, int]],
    label: int,
    k: int = 5,
    reverse: bool = True,
) -> list[tuple[int, float, Tensor, int]]:
    """
    Get the top-k results for a specific label.

    Args:
        results (list[tuple[int, float, Tensor, int]]): The list of results to filter and sort.
        label (int): The label to filter by.
        k (int, optional): The number of top results to return. Defaults to 5.
        reverse (bool, optional): Define the sorting order. If True, highest anomaly score; if False, lowest score. Defaults to True.

    Returns:
        list[tuple[int, float, Tensor, int]]: The top-k results for the specified label.
    """
    # only keep results with label equal to the specified label
    filtered = [r for r in results if r[3] == label]
    # sorted reverse by anomaly score
    sorted_result = sorted(filtered, key=lambda r: r[1], reverse=reverse)

    top_k = sorted_result[:k]
    return top_k


def main() -> None:
    """Generate anomaly figures for different methods and categories."""

    # config
    common_yaml = resolve_config_paths(config_path="common.yaml")
    models_dir = common_yaml["paths"]["models_dir"]
    figures_dir = common_yaml["paths"]["figures_dir"]
    run_id = get_run_id(tag="figure")

    # dataset
    dataset = AnomalyDataset(category=CATEGORY, split=SPLIT)

    # padim
    scores_padim = _collect_scores(
        method="padim",
        checkpoint=CHECKPOINTS["padim"],
        models_dir=models_dir,
        dataset=dataset,
    )

    top_high_normal_padim = _top_k(results=scores_padim, label=0, k=TOP_K, reverse=True)
    top_success_padim = _top_k(results=scores_padim, label=1, k=TOP_K, reverse=True)
    top_missed_anomaly_padim = _top_k(results=scores_padim, label=1, k=TOP_K, reverse=False)

    # stfpm
    scores_stfpm = _collect_scores(
        method="stfpm",
        checkpoint=CHECKPOINTS["stfpm"],
        models_dir=models_dir,
        dataset=dataset,
    )

    top_high_normal_stfpm = _top_k(results=scores_stfpm, label=0, k=TOP_K, reverse=True)
    top_success_stfpm = _top_k(results=scores_stfpm, label=1, k=TOP_K, reverse=True)
    top_missed_anomaly_stfpm = _top_k(results=scores_stfpm, label=1, k=TOP_K, reverse=False)

    # --- vizualisation ---

    score_vmax_padim = max([r[1] for r in scores_padim])
    score_vmax_stfpm = max([r[1] for r in scores_stfpm])

    # figure by model
    selection = [
        ("padim", "high_normal", top_high_normal_padim, score_vmax_padim),
        ("padim", "success", top_success_padim, score_vmax_padim),
        ("padim", "missed_anomaly", top_missed_anomaly_padim, score_vmax_padim),
        ("stfpm", "high_normal", top_high_normal_stfpm, score_vmax_stfpm),
        ("stfpm", "success", top_success_stfpm, score_vmax_stfpm),
        ("stfpm", "missed_anomaly", top_missed_anomaly_stfpm, score_vmax_stfpm),
    ]
    for model_name, result_type, top_k_results, score_vmax in selection:
        for idx, _, anomaly_map, _ in top_k_results:
            sample = dataset[idx]
            fig = generate_plots(
                image=sample["image"],
                mask=sample["mask"],
                anomaly_map=anomaly_map,
                vmin=0.0,
                vmax=score_vmax,
            )
            save_figure(
                fig=fig,
                file_path=figures_dir / run_id / model_name / result_type / f"{idx}.png",
            )

    # figure comparison
    vmax_dict = {"padim": score_vmax_padim, "stfpm": score_vmax_stfpm}
    selection_comparison = [
        ("padim", "success", top_success_padim),
        ("padim", "missed_anomaly", top_missed_anomaly_padim),
        ("stfpm", "success", top_success_stfpm),
        ("stfpm", "missed_anomaly", top_missed_anomaly_stfpm),
    ]
    for model_name, result_type, top_k_results in selection_comparison:
        extracted_idx = [r[0] for r in top_k_results]
        images = [dataset[idx]["image"] for idx in extracted_idx]
        masks = [dataset[idx]["mask"] for idx in extracted_idx]
        anomaly_maps_dict = {
            "padim": [scores_padim[idx][2] for idx in extracted_idx],
            "stfpm": [scores_stfpm[idx][2] for idx in extracted_idx],
        }
        fig = generate_comparison_plot(images=images, masks=masks, anomaly_maps=anomaly_maps_dict, vmin=0.0, vmax=vmax_dict)
        save_figure(
            fig=fig,
            file_path=figures_dir / run_id / "comparison" / f"{model_name}_{result_type}.png",
        )


if __name__ == "__main__":
    main()
