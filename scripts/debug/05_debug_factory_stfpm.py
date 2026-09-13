import torch
from _common import find_good_bad_sample, load_and_print_merge_config
from torch.utils.data import random_split

from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.models.factory import create_method

params_common_dict = load_and_print_merge_config(common_config="common.yaml", model_config="stfpm.yaml")

params_common = params_common_dict["params"]
backbone = params_common["backbone"]
device = params_common["device"]

anomaly_dataset_train = AnomalyDataset(category="bottle", split="train")

print("=============== [MODEL] -->create method ===============")

stfpm = create_method("stfpm")
print(f" [MODEL] --> Verification of type : {type(stfpm)}")


print("=============== [MODEL] --> Test fit method ===============")
# override for quick debug run, real training uses n_epochs from config
stfpm.param_epochs = 10
seed = params_common["seed"]
validation_split_ratio = params_common["validation_split_ratio"]

n_total = len(anomaly_dataset_train)
n_val = int(n_total * validation_split_ratio)
n_train = n_total - n_val
train_set, valid_set = random_split(
    anomaly_dataset_train,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(seed),
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=params_common["batch_size"],
    shuffle=True,
    num_workers=0,
    drop_last=False,
)
valid_loader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=params_common["batch_size"],
    shuffle=False,
    num_workers=0,
    drop_last=False,
)
stfpm.fit(train_loader=train_loader, val_loader=valid_loader)


print("=============== [MODEL] --> Test predict method ===============")

anomaly_dataset_test = AnomalyDataset(category="bottle", split="test")

good_image, bad_image = find_good_bad_sample(anomaly_dataset_test)

b_score_anomaly, b_anomaly_map = stfpm.predict(image=bad_image)
g_score_anomaly, g_anomaly_map = stfpm.predict(image=good_image)

print(f" [MODEL] --> Anomaly score for bad image : {b_score_anomaly}")
print(f" [MODEL] --> Anomaly map for bad image : {b_anomaly_map.shape}")
print(f" [MODEL] --> Anomaly score for good image : {g_score_anomaly}")
print(f" [MODEL] --> Anomaly map for good image : {g_anomaly_map.shape}")

print(f" [MODEL] --> Comparison of anomaly scores : {b_score_anomaly} > {g_score_anomaly}")
assert b_score_anomaly > g_score_anomaly, "Anomaly score for bad image should be greater than that for good image."
