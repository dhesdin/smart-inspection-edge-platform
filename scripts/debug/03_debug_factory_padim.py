import torch
from _common import find_good_bad_sample, load_and_print_merge_config

from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.models.factory import create_method

params_common_dict = load_and_print_merge_config(common_config="common.yaml", model_config="padim.yaml")

params_common = params_common_dict["params"]
backbone = params_common["backbone"]
device = params_common["device"]
layers = params_common["layers"]
batch_size = params_common["batch_size"]


anomaly_dataset_train = AnomalyDataset(category="bottle", split="train")

print("=============== [MODEL] -->create method ===============")

padim = create_method("padim")
print(f" [MODEL] --> Verification of type : {type(padim)}")


print("=============== [MODEL] --> Test fit method ===============")

train_loader = torch.utils.data.DataLoader(
    dataset=anomaly_dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    drop_last=False,
)
padim.fit(train_loader=train_loader)

print("=============== [MODEL] --> Test predict method ===============")

anomaly_dataset_test = AnomalyDataset(category="bottle", split="test")

good_image, bad_image = find_good_bad_sample(anomaly_dataset_test)

b_score_anomaly, b_anomaly_map = padim.predict(image=bad_image)
g_score_anomaly, g_anomaly_map = padim.predict(image=good_image)

print(f" [MODEL] --> Anomaly score for bad image : {b_score_anomaly}")
print(f" [MODEL] --> Anomaly map for bad image : {b_anomaly_map.shape}")
print(f" [MODEL] --> Anomaly score for good image : {g_score_anomaly}")
print(f" [MODEL] --> Anomaly map for good image : {g_anomaly_map.shape}")

print(f" [MODEL] --> Comparison of anomaly scores : {b_score_anomaly} > {g_score_anomaly}")
assert b_score_anomaly > g_score_anomaly, "Anomaly score for bad image should be greater than that for good image."
