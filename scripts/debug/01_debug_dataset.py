from pathlib import Path

import torch
from smart_inspection.data.dataset import AnomalyDataset
from PIL import Image
from torchvision.transforms import ToTensor

anomaly_dataset = AnomalyDataset(category="bottle", split="test")

# === ANOMALY DATASET _DISCOVER SAMPLES ===
# Print the discovered len samples
print(f"Number of samples in the dataset: {len(anomaly_dataset.samples)}")

# Total number of samples in the dataset manually import
path_to_dataset = Path("datasets/mvtec_anomaly_detection/bottle/test")
test_broken_large_samples = list(path_to_dataset.glob("broken_large/*.png"))
test_broken_small_samples = list(path_to_dataset.glob("broken_small/*.png"))
test_contamination_samples = list(path_to_dataset.glob("contamination/*.png"))
test_good_samples = list(path_to_dataset.glob("good/*.png"))

total_samples = (
    len(test_broken_large_samples)
    + len(test_broken_small_samples)
    + len(test_contamination_samples)
    + len(test_good_samples)
)
print(f"Total number of samples: {total_samples}")


count_stop_good = 0
count_stop_anomaly = 0

# print the three first anomaly sample with label 1
for sample in anomaly_dataset.samples:
    if sample["label"] == 1 and count_stop_anomaly < 3:
        print(" ==== LABEL 1 ====")
        print(f"Sample with label 1:")
        print(f"  Image Path: {sample['images_path']}")
        print(f"  Label: {sample['label']}")
        print(f"  Mask Path: {sample['mask_path']}")
        count_stop_anomaly += 1
    elif sample["label"] == 0 and count_stop_good < 3:
        print(" ==== LABEL 0 ====")
        print(f"Sample with label 0:")
        print(f"  Image Path: {sample['images_path']}")
        print(f"  Label: {sample['label']}")
        print(f"  Mask Path: {sample['mask_path']}")
        count_stop_good += 1

# === ANOMALY DATASET VERIFY IMAGE AND MASK ===

print("\n=== MASK ===")
image_mask = Image.open(anomaly_dataset.samples[0]["mask_path"])
print(f"Shape size of the first mask: {image_mask.size}")  # W,H
print(f"Shape mode of the first mask: {image_mask.mode}")  # Channel mode

image_mask = ToTensor()(image_mask)
print(f"Shape size of the first mask after ToTensor: {image_mask.shape}")  # C,H,W

print("\n=== IMAGE ===")
image = Image.open(anomaly_dataset.samples[0]["images_path"])
print(f"Shape size of the first image: {image.size}")  # W,H
print(f"Shape mode of the first image: {image.mode}")  # Channel mode

image = ToTensor()(image)
print(f"Shape size of the first image after ToTensor: {image.shape}")  #


# === ANOMALY DATASET __GETITEM__  ===
# image.shape should be (3, 256, 256), mask.shape should be (1, 256, 256), label should be 1

print("\n=== SAMPLE with label 1 ===")
sample = anomaly_dataset[0]  # label 1
print(f"Label of the sample: {sample['label']}")
print(f"Shape size of the sample image: {sample['image'].shape}")
print(f"Shape size of the sample mask: {sample['mask'].shape}")

print("\n=== SAMPLE with label 0 ===")
sample = anomaly_dataset[63]  # label 0
print(f"Label of the sample: {sample['label']}")
print(f"Shape size of the sample image: {sample['image'].shape}")
print(f"Shape size of the sample mask: {sample['mask'].shape}")
print("\n === VERIFICATION OF THE MASK ===")
print(f" Verification that the mask is a tensor of zeros {sample['mask'].sum() == 0}")

# Vérifie que les valeurs de image sont bien dans une plage cohérente avec la normalisation ImageNet (pas entre 0 et 1, mais des valeurs qui peuvent être négatives — signe que Normalize a bien été appliqué après ToTensor)
print("=== VERIFICATION OF THE IMAGE ===")
print(f"Min value of the sample image: {sample['image'].min()}")
print(f"Max value of the sample image: {sample['image'].max()}")