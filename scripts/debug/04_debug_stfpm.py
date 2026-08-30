import torch
from sklearn.model_selection import train_test_split

from smart_inspection.config.loader import merge_yaml, read_yaml, resolve_config_paths
from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.models.stfpm.model import STFPM

common_yaml_conf = resolve_config_paths(config_path="common.yaml")
stfpm_yaml_conf = read_yaml(config_path="stfpm.yaml")
params_common_dict = merge_yaml(common_config=common_yaml_conf, model_config=stfpm_yaml_conf)

params_common = params_common_dict["params"]
backbone = params_common["backbone"]
device = params_common["device"]


anomaly_dataset = AnomalyDataset(category="bottle", split="test")
stfpm = STFPM()

# Config loaded : display backbone, device of yaml
# Model : display backbone type, name, device of stfpm and real device of a parameter from stfpm

# Teacher and Student models
print(" =============== TEACHER ============== ")
print(f"[YAML] -->  Backbone : {params_common['backbone']}, device : {params_common['device']}")
print(f" [MODEL TEACHER] --> Backbone type : {type(stfpm.teacher)}, Backbone name : {stfpm.teacher.__class__.__name__}")

print(f" [MODEL TEACHER] --> Device : {stfpm.device}, Real device of a parameter : {next(stfpm.teacher.parameters()).device}")

# Freeze backbone: loop on stfpm teacher resnet params
if all(not param.requires_grad for param in stfpm.teacher.parameters()):
    print(" [MODEL TEACHER] --> Teacher backbone is frozen (requires_grad=False)")
else:
    raise AssertionError(" [MODEL TEACHER] --> Teacher backbone is NOT frozen (requires_grad=True)")

print(" =============== STUDENT ============== ")

print(f" [MODEL STUDENT] --> Backbone type : {type(stfpm.student)}, Backbone name : {stfpm.student.__class__.__name__}")
print(f" [MODEL STUDENT] --> Device : {stfpm.device}, Real device of a parameter : {next(stfpm.student.parameters()).device}")

# Freeze backbone: loop on stfpm student resnet params
if all(param.requires_grad for param in stfpm.student.parameters()):
    print(" [MODEL STUDENT] --> Student backbone is trainable (requires_grad=True)")
else:
    raise AssertionError(" [MODEL STUDENT] --> Student backbone is NOT trainable (requires_grad=False)")


# VErification between Student and Teacher model: the weights
if all(torch.equal(t_param, s_param) for t_param, s_param in zip(stfpm.teacher.parameters(), stfpm.student.parameters())):
    raise AssertionError("[WEIGHTS] --> Teacher and Student weights are the same, they should be different")
else:
    print("[WEIGHTS] --> Teacher and Student weights are different, as expected")


# Functional hooks: real image from AnomalyDataset, check that stfpm.features contains the expected keys.
image = anomaly_dataset[0]["image"]
print(f" [MODEL - IMAGE] --> Image shape : {image.shape}")  # (C,H,W)
image = image.unsqueeze(0)  # Add batch dimension
print(f" [MODEL - IMAGE] --> Image shape after unsqueeze : {image.shape}")  # (B,C,H,W)
image = image.to(stfpm.device)

# Check that the hooks have captured the features
stfpm.student(image)
stfpm.teacher(image)

print(f" [MODEL - STUDENT] --> List features keys : {list(stfpm.student_features.keys())} ")
print(f" [MODEL - TEACHER] --> List features keys : {list(stfpm.teacher_features.keys())} ")

for layer_name in stfpm.layers:
    print(f" {layer_name} : teacher: {stfpm.teacher_features[layer_name].shape}, student: {stfpm.student_features[layer_name].shape}")

print("=============== [MODEL] --> Test fit method ===============")
# ovverride for quick debug run, real training uses 100 from config
stfpm.param_epochs = 4

anomaly_dataset_train = AnomalyDataset(category="bottle", split="train")

train_set, valid_set = train_test_split(anomaly_dataset_train, test_size=0.2)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=32, shuffle=False, num_workers=0, drop_last=False)

stfpm.fit(train_loader=train_loader, val_loader=valid_loader)

print(" [MODEL] --> Finished testing fit method")

# print("=============== [MODEL] --> Test predict method ===============")
