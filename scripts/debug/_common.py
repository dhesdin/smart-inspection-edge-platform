from smart_inspection.config.loader import merge_yaml, read_yaml, resolve_config_paths


def load_and_print_common_config(config_path: str):
    common_yaml_conf = resolve_config_paths(config_path=str(config_path))
    if not common_yaml_conf:
        raise FileNotFoundError("Common configuration file not found.")
    print(" Common configuration yaml:")
    params_common = common_yaml_conf["params"]
    for key, value in params_common.items():
        print(f"{key}: {value}")


def load_and_print_merge_config(common_config: str, model_config: str) -> dict:
    common_yaml_conf = resolve_config_paths(config_path=common_config)
    if not common_yaml_conf:
        raise FileNotFoundError("Common configuration file not found.")
    model_yaml_conf = read_yaml(config_path=model_config)
    if not model_yaml_conf:
        raise FileNotFoundError("Model configuration file not found.")
    merge_conf_yaml = merge_yaml(common_config=common_yaml_conf, model_config=model_yaml_conf)
    merge_conf_params = merge_conf_yaml["params"]
    if not merge_conf_params:
        raise ValueError("Merged configuration 'params' section is empty or missing.")

    for k, v in merge_conf_params.items():
        print(f"{k}: {v}")
    return merge_conf_yaml


def find_good_bad_sample(dataset):
    good_image = None
    bad_image = None

    for i in range(len(dataset)):
        sample = dataset[i]

        if sample["label"] == 0 and good_image is None:
            good_image = sample["image"]
            print(f"[MODEL] --> Good image found at index {i}")

        elif sample["label"] == 1 and bad_image is None:
            bad_image = sample["image"]
            print(f"[MODEL] --> Bad image found at index {i}")

        if good_image is not None and bad_image is not None:
            break

    assert good_image is not None, "No good image found in the dataset."
    assert bad_image is not None, "No bad image found in the dataset."

    print(f"[MODEL] --> Good image shape : {good_image.shape}")
    print(f"[MODEL] --> Bad image shape : {bad_image.shape}")

    return good_image, bad_image
