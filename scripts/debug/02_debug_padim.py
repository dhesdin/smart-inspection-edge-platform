import torch
from _common import find_good_bad_sample, load_and_print_merge_config

from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.models.padim.model import PaDiM

params_common_dict = load_and_print_merge_config(common_config="common.yaml", model_config="padim.yaml")

params_common = params_common_dict["params"]
backbone = params_common["backbone"]
device = params_common["device"]
layers = params_common["layers"]
batch_size = params_common["batch_size"]


anomaly_dataset = AnomalyDataset(category="bottle", split="test")
padim = PaDiM()

# Config loaded : display backbone, layers, device of yaml
# Model : display backbone type, name, device of padim and real device of a parameter from padim
print(f"[YAML] -->  Backbone : {params_common['backbone']}, layers : {params_common['layers']}, device : {params_common['device']}")
print(f" [MODEL] --> Backbone type : {type(padim.resnet)}, Backbone name : {padim.resnet.__class__.__name__}")
print(f" [MODEL] --> Device : {padim.device}, Real device of a parameter : {next(padim.resnet.parameters()).device}")

# Freeze backbone: loop on padim resnet params
if all(not param.requires_grad for param in padim.resnet.parameters()):
    print(" [MODEL] --> Backbone is frozen (requires_grad=False)")
else:
    raise AssertionError(" [MODEL] --> Backbone is NOT frozen (requires_grad=True)")
# Eval mode : resnet training should eb false

if not padim.resnet.training:
    print(" [MODEL] --> Backbone is in eval mode (training=False)")
else:
    raise AssertionError(" [MODEL] --> Backbone is NOT in eval mode (training=True)")
# Functional hooks: real image from AnomalyDataset, check that padim.features contains the expected keys.

image = anomaly_dataset[0]["image"]  # Tensor of shape (C,H,W)
print(f" [MODEL] --> Image shape before unsqueeze: {image.shape}")
image = image.unsqueeze(0)  # Add batch dimension (B(1),C,H,W)
print(f" [MODEL] --> Image shape after unsqueeze: {image.shape}")
image = image.to(padim.device)
padim.resnet(image)  # forward pass to trigger hooks

print(f" [MODEL] --> List of feature keys: {list(padim.features.keys())}")


print("=============== [MODEL] --> Test fit method ===============")
anomaly_dataset_train = AnomalyDataset(category="bottle", split="train")

train_loader = torch.utils.data.DataLoader(
    dataset=anomaly_dataset_train,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
    drop_last=False,
)

padim.fit(train_loader=train_loader)
print(f" [MODEL] --> Embeddings shape: {padim.embeddings.shape}")
print(f" [MODEL] --> Layers : {list(padim.features.keys())}")
print(f" [MODEL] --> Mean shape: {padim.mean.shape}")
print(f" [MODEL] --> Cov shape: {padim.cov.shape}")
# print(f" [MODEL] --> FFirst 2 values of mean: {padim.mean[:2]}")
# print(f" [MODEL] --> FFirst 2 values of cov: {padim.cov[:2]}")

print("=============== [MODEL] --> Test predict method ===============")

good_image, bad_image = find_good_bad_sample(anomaly_dataset)

b_score_anomaly, b_anomaly_map = padim.predict(image=bad_image)
g_score_anomaly, g_anomaly_map = padim.predict(image=good_image)


print(f" [MODEL] [BAD_IMAGE]--> Score Anomaly : {b_score_anomaly}")
print(f" [MODEL] [BAD_IMAGE]--> Anomaly Map shape: {b_anomaly_map.shape}")
print(f" [MODEL] [GOOD_IMAGE]--> Score Anomaly : {g_score_anomaly}")
print(f" [MODEL] [GOOD_IMAGE]--> Anomaly Map shape: {g_anomaly_map.shape}")
