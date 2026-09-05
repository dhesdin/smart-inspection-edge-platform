import copy

import torch

from smart_inspection.config.loader import merge_yaml, read_yaml, resolve_config_paths
from smart_inspection.data.dataset import AnomalyDataset
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
validation_split_ratio = stfpm_common_dict["params"]["validation_split_ratio"]

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

# ======================================== #
#                   STFPM                  #
# ======================================== #


print("=============== [MODEL] --> create method ===============")

stfpm = create_method("stfpm")
dataset = AnomalyDataset(category="bottle", split="train")


assert stfpm.validation_split_ratio is not None and stfpm.validation_split_ratio == validation_split_ratio
print(f"Validation split ratio for STFPM: {stfpm.validation_split_ratio}")

print("=============== [MODEL STFPM] --> Test  training method ===============")
# override for quick debug run, real training uses n_epochs from config
stfpm.param_epochs = 10
student_before = copy.deepcopy(stfpm.student.state_dict())
train(method=stfpm, dataset=dataset, batch_size=batch_size, seed=seed, shuffle=True, num_workers=2, drop_last=True)


student_after = stfpm.student.state_dict()

weights_changed = [not torch.equal(student_before[key], student_after[key]) for key in student_before]

n_changed = sum(weights_changed)
n_total = len(weights_changed)

ratio_changed = n_changed / n_total if n_total > 0 else 0
print(f" ratio of changed weights : {ratio_changed}")

assert ratio_changed > 0.5
# ======================================== #
#                   PaDiM                  #
# ======================================== #
print("=============== [MODEL PaDiM] --> Test  training method ===============")
padim = create_method("padim")


assert padim.validation_split_ratio is None
print("PaDiM not using a validation split.")


train(method=padim, dataset=dataset, batch_size=batch_size, seed=seed, shuffle=True, num_workers=2, drop_last=False)

assert padim.mean is not None and padim.mean.shape[1] == padim_yaml_conf["params"]["n_features"]
assert padim.cov is not None
assert padim.selected_indices is not None and len(padim.selected_indices) == padim_yaml_conf["params"]["n_features"]

print("PaDiM internal state changed correctly.")
