from pathlib import Path
from smart_inspection.data.dataset import AnomalyDataset



anomaly_dataset = AnomalyDataset(category="bottle", split="test")

# Print the discovered len samples
print(f"Number of samples in the dataset: {len(anomaly_dataset.samples)}")

# Total number of samples in the dataset manually import
path_to_dataset = Path("datasets/mvtec_anomaly_detection/bottle/test")
test_broken_large_samples = list(path_to_dataset.glob("broken_large/*.png"))
test_broken_small_samples = list(path_to_dataset.glob("broken_small/*.png"))
test_contamination_samples = list(path_to_dataset.glob("contamination/*.png"))
test_good_samples = list(path_to_dataset.glob("good/*.png"))

total_samples = len(test_broken_large_samples) + len(test_broken_small_samples) + len(test_contamination_samples) + len(test_good_samples)
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
        count_stop_anomaly +=1
    elif sample["label"] == 0 and count_stop_good < 3:
        print(" ==== LABEL 0 ====")
        print(f"Sample with label 0:")
        print(f"  Image Path: {sample['images_path']}")
        print(f"  Label: {sample['label']}")
        print(f"  Mask Path: {sample['mask_path']}")
        count_stop_good +=1
