import numpy as np
import torch

from smart_inspection.config.loader import resolve_config_paths
from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.data.splits import split_data
from smart_inspection.evaluation.evaluate import evaluate
from smart_inspection.evaluation.scores import collect_scores
from smart_inspection.evaluation.threshold import compute_metrics, compute_threshold
from smart_inspection.io.run_id import get_run_id
from smart_inspection.io.save_json import save_json
from smart_inspection.models.factory import create_method
from smart_inspection.training.train import train

# CONSTANTS
CATEGORY_DATASET = "cable"


def main() -> None:
    """Run the threshold protocol."""

    # config parameters
    common_yaml = resolve_config_paths(config_path="common.yaml")
    batch_size = common_yaml["params"]["batch_size"]
    seed = common_yaml["params"]["seed"]
    calibration_ratio = common_yaml["params"]["calibration_ratio"]
    validation_ratio = common_yaml["params"]["validation_ratio"]
    threshold_percentile = common_yaml["params"]["threshold_percentile"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = get_run_id(tag=f"threshold_protocol_{device}")
    reports_dir = common_yaml["paths"]["reports_dir"]
    models_dir = common_yaml["paths"]["models_dir"]

    # dataset
    train_good_dataset = AnomalyDataset(category=CATEGORY_DATASET, split="train")
    test_dataset = AnomalyDataset(category=CATEGORY_DATASET, split="test")

    calibration_idx, validation_idx, training_idx = split_data(
        n_images=len(train_good_dataset),
        calibration_ratio=calibration_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    calibration_dataset = torch.utils.data.Subset(train_good_dataset, calibration_idx)
    validation_dataset = torch.utils.data.Subset(train_good_dataset, validation_idx)
    training_dataset = torch.utils.data.Subset(train_good_dataset, training_idx)

    # DataLoaders (the train and validation loader is in train function)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # create and train models
    results = {}
    for method_name in ["padim", "stfpm"]:
        model = create_method(method_name=method_name)
        checkpoint_stem = f"{method_name}_{CATEGORY_DATASET}_{run_id}"
        checkpoint = models_dir / f"{checkpoint_stem}.pt"
        train(
            method=model,
            train_dataset=training_dataset,
            val_dataset=validation_dataset,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            save_path=checkpoint,
        )
        # collect score only from the calibration to compute the threshold
        calibration_score, _ = collect_scores(method=model, dataset=calibration_dataset)
        threshold = compute_threshold(scores=calibration_score, percentile=threshold_percentile)
        # stats on the calibration scores used to fix the threshold, so the decision can be audited
        calibration_score_array = np.asarray(calibration_score, dtype=float)
        calibration_stats = {
            "n": int(calibration_score_array.size),
            "min": float(calibration_score_array.min()),
            "median": float(np.median(calibration_score_array)),
            "max": float(calibration_score_array.max()),
            "n_scores_ge_threshold": int(np.sum(calibration_score_array >= threshold)),
        }
        # collect test scores and compute metrics using the threshold
        test_scores, test_labels = collect_scores(method=model, dataset=test_dataset)
        metrics = compute_metrics(scores=test_scores, labels=test_labels, threshold=threshold)

        # evaluate the metrics and store the results
        evaluation = evaluate(method=model, test_loader=test_dataloader)
        # storage
        results[method_name] = {
            "checkpoint": checkpoint_stem,
            "threshold": threshold,
            "calibration_stats": calibration_stats,
            "metrics": metrics,
            "evaluation": evaluation,
        }
        del model
        torch.cuda.empty_cache()

    config = {
        "run_id": run_id,
        "dataset": CATEGORY_DATASET,
        "split": {
            "n_train_good": len(train_good_dataset),
            "n_calibration": len(calibration_idx),
            "n_validation": len(validation_idx),
            "n_training": len(training_idx),
            "calibration_idx": calibration_idx,
            "validation_idx": validation_idx,
            "training_idx": training_idx,
        },
        "config": {
            "calibration_ratio": calibration_ratio,
            "validation_ratio": validation_ratio,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "seed": seed,
            "batch_size": batch_size,
            "threshold_percentile": threshold_percentile,
            "percentile_method": "linear",
            "decision_rule": "score >= T",
        },
        "results": results,
    }
    save_json(data=config, file_path=reports_dir / f"results_{run_id}.json")


if __name__ == "__main__":
    main()
