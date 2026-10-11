import gc
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smart_inspection.config.loader import resolve_config_paths
from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.evaluation.evaluate import evaluate
from smart_inspection.evaluation.reference_check import compare_with_reference
from smart_inspection.io.run_id import get_run_id
from smart_inspection.io.save_json import save_json
from smart_inspection.models.factory import create_method

# CONSTANTS
CHECKPOINTS = {
    "stfpm": "stfpm_cable_cuda_20261003_171414",
    "padim": "padim_cable_cuda_20261003_171414",
}
REFERENCE_REPORT = "results_compare_padim_stfpm_cuda_20261003_171414"


def _evaluate_one(method: str, checkpoint: str, models_dir: Path, test_loader: DataLoader) -> dict:
    """
    Evaluate a single method from its checkpoint on the test dataset.

    Args:
        method (str): The name of the method to evaluate.
        checkpoint (str): The checkpoint file name (without extension) to load the model from.
        models_dir (Path): The directory containing the model checkpoints.
        test_loader (DataLoader): The DataLoader for the test dataset.

    Returns:
        dict: The evaluation results.
    """
    if not checkpoint:
        raise ValueError(f"No checkpoint specified for method '{method}'")
    if not (models_dir / f"{checkpoint}.pt").exists():
        raise FileNotFoundError(f"Checkpoint file '{checkpoint}.pt' for method '{method}' not found in '{models_dir}'")

    method_instance = create_method(method_name=method)
    method_instance.load(models_dir / f"{checkpoint}.pt")

    evaluate_result = evaluate(method=method_instance, test_loader=test_loader)
    # Delete the method instance to free up memory between calls
    del method_instance
    gc.collect()  # garbage collection for freeing up memory after deleting unused ref
    torch.cuda.empty_cache()  # clear GPU mem cache

    return evaluate_result


def main() -> None:
    """Compute the image-level AUROC per defect type of each method, from its saved checkpoint.

    The checkpoints are first checked against the reference report (same counts, image AUROC within tolerance).
    The report is always saved with this check, then the script fails if the check does not pass.
    """

    # config
    common_yaml = resolve_config_paths(config_path="common.yaml")
    reports_dir = common_yaml["paths"]["reports_dir"]
    models_dir = common_yaml["paths"]["models_dir"]
    batch_size = common_yaml["params"]["batch_size"]
    seed = common_yaml["params"]["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = get_run_id(tag=f"compare_method_bysubcategory_{device}")

    # dataset
    category_dataset = "cable"
    test_dataset = AnomalyDataset(category=category_dataset, split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # REFERENCE REPORT RESULTS
    reference_report = reports_dir / f"{REFERENCE_REPORT}.json"
    if not reference_report.exists():
        raise FileNotFoundError(f"Reference report '{REFERENCE_REPORT}.json' not found in '{reports_dir}'")
    with open(reference_report, "r") as f:
        reference_data = json.load(f)
    if reference_data["config"]["dataset"] != category_dataset:
        raise ValueError(f"Reference report is on '{reference_data['config']['dataset']}', expected '{category_dataset}'")

    # Evaluate STFPM and PaDiM methods
    evaluate_result_stfpm = _evaluate_one(
        method="stfpm",
        checkpoint=CHECKPOINTS["stfpm"],
        models_dir=models_dir,
        test_loader=test_loader,
    )
    print(f"STFPM evaluation result: {evaluate_result_stfpm}")

    evaluate_result_padim = _evaluate_one(
        method="padim",
        checkpoint=CHECKPOINTS["padim"],
        models_dir=models_dir,
        test_loader=test_loader,
    )
    print(f"PaDiM evaluation result: {evaluate_result_padim}")

    # ------ Compare evaluation results with reference values ------

    reference_device = reference_data["config"]["device"]
    check_stfpm = compare_with_reference(evaluate_result_stfpm, reference_data["stfpm"], device, reference_device)
    check_padim = compare_with_reference(evaluate_result_padim, reference_data["padim"], device, reference_device)
    # Global verdict based on counts match and gaps within tolerance (device_match is informational only)
    verdict = all(check["counts_match"] and check["image_within_tolerance"] for check in (check_stfpm, check_padim))

    # report
    config = {
        "reference_file_name": reference_report.name,
        "run_id": run_id,
        "dataset_category": category_dataset,
        "checkpoints": CHECKPOINTS,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "n_images": len(test_dataset),
    }
    reference_check = {
        "verdict": "pass" if verdict else "fail",
        "stfpm": check_stfpm,
        "padim": check_padim,
    }
    report = {
        "config": config,
        "reference_check": reference_check,
        "stfpm": evaluate_result_stfpm,
        "padim": evaluate_result_padim,
    }
    save_json(data=report, file_path=reports_dir / f"results_{run_id}.json")

    # raised after the report is saved, so the diagnostic is kept
    if not verdict:
        failures = []
        for method, check in (("STFPM", check_stfpm), ("PaDiM", check_padim)):
            if not check["counts_match"]:
                failures.append(f"{method}: test set counts differ from the reference report, the tolerance does not apply.")
            if not check["image_within_tolerance"]:
                failures.append(f"{method}: image-level AUROC gap {check['gap_image_auroc']} beyond tolerance {check['tolerance_image_auroc']}.")
        raise ValueError("Checkpoints do not reproduce the reference report:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
