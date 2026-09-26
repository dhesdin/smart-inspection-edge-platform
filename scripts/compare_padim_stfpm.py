import torch

from smart_inspection.config.loader import resolve_config_paths
from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.evaluation.evaluate import evaluate
from smart_inspection.io.run_id import get_run_id
from smart_inspection.io.save_json import save_json
from smart_inspection.models.factory import create_method
from smart_inspection.training.train import train


def main() -> None:
    """Compare PaDiM and STFPM models."""

    # config
    common_yaml = resolve_config_paths(config_path="common.yaml")
    batch_size = common_yaml["params"]["batch_size"]
    seed = common_yaml["params"]["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = get_run_id(tag=f"compare_padim_stfpm_{device}")
    run_id_only_date = get_run_id(tag=f"{device}")
    reports_dir = common_yaml["paths"]["reports_dir"]
    models_dir = common_yaml["paths"]["models_dir"]

    # dataset
    category_dataset = "cable"
    train_dataset = AnomalyDataset(category=category_dataset, split="train")
    test_dataset = AnomalyDataset(category=category_dataset, split="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # dict to store evaluation results
    results = {}

    # STFPM model train + eval
    print("Training STFPM model...")
    stfpm_model = create_method(method_name="stfpm")
    train(
        method=stfpm_model,
        dataset=train_dataset,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        save_path=models_dir / f"stfpm_{category_dataset}_{run_id_only_date}.pt",
    )
    print("[STFPM] Training completed. Starting evaluation...")
    results["stfpm"] = evaluate(method=stfpm_model, test_loader=test_loader)
    print(f"[STFPM] AUROC: {results['stfpm']['roc_auc']:.4f}")
    # clear GPU memory
    print("Clearing GPU memory...")
    del stfpm_model  # delete STFPM model to move gpu memory in cache
    torch.cuda.empty_cache()  # clear gpu memory after that

    # PaDiM model train + eval
    print("Training PaDiM model...")
    padim_model = create_method(method_name="padim")
    train(
        method=padim_model,
        dataset=train_dataset,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        save_path=models_dir / f"padim_{category_dataset}_{run_id_only_date}.pt",
    )
    print("[PaDiM] Training completed. Starting evaluation...")
    results["padim"] = evaluate(method=padim_model, test_loader=test_loader)
    print(f"[PaDiM] AUROC: {results['padim']['roc_auc']:.4f}")

    # save results
    config = {
        "run_id": run_id,
        "dataset": category_dataset,
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
    }
    report = {"config": config, **results}
    save_json(data=report, file_path=reports_dir / f"results_{run_id}.json")
    print(f"Trainings of models completed. Models saved to {models_dir}")


if __name__ == "__main__":
    main()
