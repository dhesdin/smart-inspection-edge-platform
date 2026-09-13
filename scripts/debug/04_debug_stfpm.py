import torch
from _common import find_good_bad_sample, load_and_print_merge_config
from torch.utils.data import random_split

from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.models.stfpm.model import STFPM

params_common_dict = load_and_print_merge_config(common_config="common.yaml", model_config="stfpm.yaml")

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

print("\n=============== [MODEL] --> Test fit method ===============")
# ovverride for quick debug run, real training uses 100 from config
stfpm.param_epochs = 10
seed = params_common["seed"]
validation_split_ratio = params_common["validation_split_ratio"]
batch_size = params_common["batch_size"]
anomaly_dataset_train = AnomalyDataset(category="bottle", split="train")

n_total = len(anomaly_dataset_train)
n_val = int(n_total * validation_split_ratio)
n_train = n_total - n_val
train_set, valid_set = random_split(
    anomaly_dataset_train,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(seed),
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)


stfpm.fit(train_loader=train_loader, val_loader=valid_loader)

print(" [MODEL] --> Finished testing fit method\n")


print("=============== [MODEL] --> EVALUATE : BATCH LABEL CHECK ===============")
for batch in train_loader:
    print(f"Type batch['label'][0] : {type(batch['label'][0])}")
    print(f"batch['label'][0] : {batch['label'][0]}")
    break
print("\n=============== [MODEL] --> Test predict method ===============")

good_image, bad_image = find_good_bad_sample(anomaly_dataset)

b_score_anomaly, b_anomaly_map = stfpm.predict(image=bad_image)
g_score_anomaly, g_anomaly_map = stfpm.predict(image=good_image)
print(f"[MODEL] --> Bad image anomaly score : {b_score_anomaly}" + f" | Bad image anomaly map shape : {b_anomaly_map.shape}")
print(f"[MODEL] --> Good image anomaly score : {g_score_anomaly}" + f" | Good image anomaly map shape : {g_anomaly_map.shape}")
