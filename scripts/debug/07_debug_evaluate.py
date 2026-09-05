import torch

from smart_inspection.config.loader import merge_yaml, read_yaml, resolve_config_paths
from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.evaluation.evaluate import evaluate
from smart_inspection.models.factory import create_method
from smart_inspection.training.train import train

common_yaml_conf = resolve_config_paths(config_path="common.yaml")
stfpm_yaml_conf = read_yaml(config_path="stfpm.yaml")
padim_yaml_conf = read_yaml(config_path="padim.yaml")
stfpm_common_dict = merge_yaml(common_config=common_yaml_conf, model_config=stfpm_yaml_conf)
padim_common_dict = merge_yaml(common_config=common_yaml_conf, model_config=padim_yaml_conf)

# param from original common yaml
params_common = common_yaml_conf["params"]

seed = params_common["seed"]
batch_size = params_common["batch_size"]

# verifications of all params
print("=============== [PARAMS] --> Common ===============")
for key, value in params_common.items():
    print(f"{key}: {value}")

print("=============== [PARAMS] --> STFPM ===============")
for key, value in stfpm_common_dict["params"].items():
    print(f"{key}: {value}")

print("=============== [PARAMS] --> PaDiM ===============")
for key, value in padim_common_dict["params"].items():
    print(f"{key}: {value}")

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
