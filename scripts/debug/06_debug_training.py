import copy

import torch
from _common import load_and_print_merge_config

from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.models.factory import create_method
from smart_inspection.training.train import train

print("=============== [PARAMS] --> STFPM ===============")
stfpm_common_dict = load_and_print_merge_config(common_config="common.yaml", model_config="stfpm.yaml")

print("=============== [PARAMS] --> PaDiM ===============")
padim_common_dict = load_and_print_merge_config(common_config="common.yaml", model_config="padim.yaml")

seed = stfpm_common_dict["params"]["seed"]
batch_size = stfpm_common_dict["params"]["batch_size"]
validation_split_ratio = stfpm_common_dict["params"]["validation_split_ratio"]

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

assert padim.mean is not None and padim.mean.shape[1] == padim_common_dict["params"]["n_features"]
assert padim.cov is not None
assert padim.selected_indices is not None and len(padim.selected_indices) == padim_common_dict["params"]["n_features"]

print("PaDiM internal state changed correctly.")
