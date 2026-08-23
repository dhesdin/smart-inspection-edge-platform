from pathlib import Path

import torch
from smart_inspection.config.loader import merge_yaml, read_yaml, resolve_config_paths
from smart_inspection.data.dataset import AnomalyDataset
from PIL import Image
from torchvision.transforms import ToTensor

from smart_inspection.models.stfpm.model import STFPM

common_yaml_conf = resolve_config_paths(config_path="common.yaml")
stfpm_yaml_conf = read_yaml(config_path="stfpm.yaml")
params_common_dict = merge_yaml(
    common_config=common_yaml_conf, model_config=stfpm_yaml_conf
)

params_common = params_common_dict["params"]
backbone = params_common["backbone"]
device = params_common["device"]


anomaly_dataset = AnomalyDataset(category="bottle", split="test")
stfpm = STFPM()

# Config loaded : display backbone, device of yaml
# Model : display backbone type, name, device of stfpm and real device of a parameter from stfpm

# Teacher and Student models
print(" =============== TEACHER ============== ")
print(
    f"[YAML] -->  Backbone : {params_common['backbone']}, device : {params_common['device']}"
)
print(
    f" [MODEL TEACHER] --> Backbone type : {type(stfpm.teacher)}, Backbone name : {stfpm.teacher.__class__.__name__}"
)

print(
    f" [MODEL TEACHER] --> Device : {stfpm.device}, Real device of a parameter : {next(stfpm.teacher.parameters()).device}"
)

# Freeze backbone: loop on stfpm teacher resnet params
if all(not param.requires_grad for param in stfpm.teacher.parameters()):
    print(" [MODEL TEACHER] --> Teacher backbone is frozen (requires_grad=False)")
else:
    raise AssertionError(
        " [MODEL TEACHER] --> Teacher backbone is NOT frozen (requires_grad=True)"
    )

print(" =============== STUDENT ============== ")

print(
    f" [MODEL STUDENT] --> Backbone type : {type(stfpm.student)}, Backbone name : {stfpm.student.__class__.__name__}"
)
print(
    f" [MODEL STUDENT] --> Device : {stfpm.device}, Real device of a parameter : {next(stfpm.student.parameters()).device}"
)

# Freeze backbone: loop on stfpm student resnet params
if all(param.requires_grad for param in stfpm.student.parameters()):
    print(" [MODEL STUDENT] --> Student backbone is trainable (requires_grad=True)")
else:
    raise AssertionError(
        " [MODEL STUDENT] --> Student backbone is NOT trainable (requires_grad=False)"
    )


# VErification between Student and Teacher model: the weights
if all(
    torch.equal(t_param, s_param)
    for t_param, s_param in zip(stfpm.teacher.parameters(), stfpm.student.parameters())
):
    raise AssertionError(
        "[WEIGHTS] --> Teacher and Student weights are the same, they should be different"
    )
else:
    print("[WEIGHTS] --> Teacher and Student weights are different, as expected")

# print("=============== [MODEL] --> Test fit method ===============")

# print("=============== [MODEL] --> Test predict method ===============")
