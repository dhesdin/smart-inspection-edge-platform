# from smart_inspection.data.dataset import AnomalyDataset
from smart_inspection.config.loader import read_yaml, resolve_config_paths, merge_yaml, get_root_dir



root = get_root_dir()
print("Root directory:", root)

common_yaml= resolve_config_paths(config_path="common.yaml")

read_yaml_common = read_yaml(config_path="common.yaml")
read_yaml_padim = read_yaml(config_path="padim.yaml")

print("================ COMMON YAML ===============")
print("Common YAML:\n", read_yaml_common)
print("================ PADIM YAML ===============")
print("Padim YAML:\n", read_yaml_padim)

merged_yaml = merge_yaml(common_config=common_yaml, model_config=read_yaml_padim)

print("================ Merged YAML ===============")
print("Merged YAML:\n", merged_yaml)

# tmp_datasets = [
#     {"path": "datasets/mvtec_anomaly_detection/bottle/test/good/000.png", "label": 0, "mask_path": None},
#     {"path": "datasets/mvtec_anomaly_detection/bottle/test/good/001.png", "label": 0, "mask_path": None},
#     {"path": "datasets/mvtec_anomaly_detection/bottle/test/broken_large/000.png", "label": 1, "mask_path": "datasets/mvtec_anomaly_detection/bottle/ground_truth/broken_large/000_mask.png"},
# ]

# dataset = AnomalyDataset()