import torch
from _common import load_and_print_merge_config

from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.evaluation.evaluate import evaluate
from smart_inspection.models.factory import create_method
from smart_inspection.training.train import train

print("=============== [PARAMS] --> STFPM ===============")
stfpm_common_dict = load_and_print_merge_config(common_config="common.yaml", model_config="stfpm.yaml")

print("=============== [PARAMS] --> PaDiM ===============")
padim_common_dict = load_and_print_merge_config(common_config="common.yaml", model_config="padim.yaml")

seed = stfpm_common_dict["params"]["seed"]
batch_size = stfpm_common_dict["params"]["batch_size"]

# datasets
train_dataset = AnomalyDataset(category="bottle", split="train")
test_dataset = AnomalyDataset(category="bottle", split="test")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("=============== [MODEL STFPM] --> Test  evaluate method ===============")

# ======================================== #
#                   STFPM                  #
# ======================================== #
stfpm = create_method("stfpm")
stfpm.param_epochs = 10  # fast run for debug

train(method=stfpm, dataset=train_dataset, batch_size=batch_size, seed=seed, shuffle=True, num_workers=2, drop_last=True)

results_stfpm = evaluate(method=stfpm, test_loader=test_loader)

print(f"MODEL STFPM: RESULTS: {results_stfpm}")
assert results_stfpm["roc_auc"] > 0.7


print("\n=============== [MODEL PaDiM] --> Test  evaluate method ===============")
# ======================================== #
#                   PaDiM                  #
# ======================================== #
padim = create_method("padim")

train(method=padim, dataset=train_dataset, batch_size=batch_size, seed=seed, shuffle=True, num_workers=2, drop_last=False)

results_padim = evaluate(method=padim, test_loader=test_loader)

print(f"MODEL PaDiM: RESULTS: {results_padim}")
assert results_padim["roc_auc"] > 0.7
