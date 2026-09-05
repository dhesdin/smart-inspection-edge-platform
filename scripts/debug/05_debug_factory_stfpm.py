import torch
from torch.utils.data import random_split

from smart_inspection.config.loader import merge_yaml, read_yaml, resolve_config_paths
from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.models.factory import create_method

common_yaml_conf = resolve_config_paths(config_path="common.yaml")
stfpm_yaml_conf = read_yaml(config_path="stfpm.yaml")
params_common_dict = merge_yaml(common_config=common_yaml_conf, model_config=stfpm_yaml_conf)

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

good_image = None
bad_image = None
# find good label and not good label
for i in range(len(anomaly_dataset_test)):
    sample = anomaly_dataset_test[i]

    if sample["label"] == 0 and good_image is None:
        good_image = sample["image"]
        print(f"[MODEL] --> Good image found at index {i}")

    elif sample["label"] == 1 and bad_image is None:
        bad_image = sample["image"]
        print(f"[MODEL] --> Bad image found at index {i}")

    if good_image is not None and bad_image is not None:
        break


print(f" [MODEL] --> Bad image shape : {bad_image.shape}")
print(f" [MODEL] --> Good image shape : {good_image.shape}")

assert good_image is not None, "No good image found in the test dataset."
assert bad_image is not None, "No bad image found in the test dataset."


b_score_anomaly, b_anomaly_map = stfpm.predict(image=bad_image)
g_score_anomaly, g_anomaly_map = stfpm.predict(image=good_image)

print(f" [MODEL] --> Anomaly score for bad image : {b_score_anomaly}")
print(f" [MODEL] --> Anomaly map for bad image : {b_anomaly_map.shape}")
print(f" [MODEL] --> Anomaly score for good image : {g_score_anomaly}")
print(f" [MODEL] --> Anomaly map for good image : {g_anomaly_map.shape}")

print(f" [MODEL] --> Comparison of anomaly scores : {b_score_anomaly} > {g_score_anomaly}")
assert b_score_anomaly > g_score_anomaly, "Anomaly score for bad image should be greater than that for good image."
