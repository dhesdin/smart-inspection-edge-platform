from pathlib import Path

import torch
from smart_inspection.config.loader import merge_yaml, read_yaml, resolve_config_paths
from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.models.padim.model import PaDiM
from PIL import Image
from torchvision.transforms import ToTensor

common_yaml_conf = resolve_config_paths(config_path="common.yaml")
padim_yaml_conf = read_yaml(config_path="padim.yaml")
params_common_dict = merge_yaml(
    common_config=common_yaml_conf, model_config=padim_yaml_conf
)

params_common = params_common_dict["params"]
backbone = params_common["backbone"]
device = params_common["device"]
layers = params_common["layers"]
batch_size = params_common["batch_size"]


anomaly_dataset = AnomalyDataset(category="bottle", split="test")
padim = PaDiM()

# Config loaded : display backbone, layers, device of yaml
# Model : display backbone type, name, device of padim and real device of a parameter from padim
print(
    f"[YAML] -->  Backbone : {params_common['backbone']}, layers : {params_common['layers']}, device : {params_common['device']}"
)
print(
    f" [MODEL] --> Backbone type : {type(padim.resnet)}, Backbone name : {padim.resnet.__class__.__name__}"
)
print(
    f" [MODEL] --> Device : {padim.device}, Real device of a parameter : {next(padim.resnet.parameters()).device}"
)

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
print(f" [MODEL] --> FFirst 2 values of mean: {padim.mean[:2]}")
print(f" [MODEL] --> FFirst 2 values of cov: {padim.cov[:2]}")
