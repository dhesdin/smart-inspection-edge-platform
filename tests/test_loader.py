import pytest
import yaml

from smart_inspection.config import loader
from smart_inspection.config.loader import get_root_dir, merge_yaml, read_yaml, resolve_config_paths

# ---------------- get_root_dir ---------------- #


# The project root is the directory that contains pyproject.toml.
def test_get_root_dir_contains_pyproject():
    assert (get_root_dir() / "pyproject.toml").exists()


# ---------------- read_yaml ---------------- #


# The real common.yaml is read into a dict with the expected sections.
def test_read_yaml_reads_common_config():
    config = read_yaml("common.yaml")

    assert {"data", "params", "paths"} <= set(config.keys())


# A config file that does not exist raises FileNotFoundError.
def test_read_yaml_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_yaml("this_file_does_not_exist.yaml")


# A syntactically broken YAML file raises YAMLError.
def test_read_yaml_broken_file_raises_yaml_error(tmp_path, monkeypatch):
    broken = tmp_path / "broken.yaml"
    broken.write_text("params: [unclosed")
    monkeypatch.setattr(loader, "_get_config_path", lambda config_path: broken)

    with pytest.raises(yaml.YAMLError, match="broken"):
        read_yaml("broken.yaml")


# ---------------- resolve_config_paths ---------------- #


# "paths" and "data" entries are resolved against the project root, other sections are left untouched.
def test_resolve_config_paths_resolves_only_paths_and_data(tmp_path, monkeypatch):
    fake_config = {
        "paths": {"models_dir": "outputs/models"},
        "data": {"datasets": "datasets/mvtec"},
        "params": {"seed": 42, "batch_size": 32},
    }
    monkeypatch.setattr(loader, "read_yaml", lambda config_path: fake_config)
    monkeypatch.setattr(loader, "get_root_dir", lambda: tmp_path)

    config = resolve_config_paths("whatever.yaml")

    assert config["paths"]["models_dir"] == tmp_path / "outputs" / "models"
    assert config["data"]["datasets"] == tmp_path / "datasets" / "mvtec"
    assert config["params"] == {"seed": 42, "batch_size": 32}


# On the real common.yaml, every path is absolute and lives under the project root.
def test_resolve_config_paths_real_config_is_under_root():
    config = resolve_config_paths("common.yaml")

    for path in config["paths"].values():
        assert path.is_absolute()
        assert get_root_dir() in path.parents


# ---------------- merge_yaml ---------------- #


# For a section present in both configs, model values override common ones and other keys are kept.
def test_merge_yaml_model_overrides_common_within_a_section():
    common = {"params": {"a": 1, "b": 2}}
    model = {"params": {"b": 3}}

    assert merge_yaml(common, model) == {"params": {"a": 1, "b": 3}}


# Sections only in common are kept, sections only in model are added.
def test_merge_yaml_keeps_and_adds_exclusive_sections():
    common = {"paths": {"x": "1"}}
    model = {"model": {"y": 2}}

    assert merge_yaml(common, model) == {"paths": {"x": "1"}, "model": {"y": 2}}


# The input dictionaries are not modified by the merge.
def test_merge_yaml_does_not_mutate_inputs():
    common = {"params": {"a": 1}}
    model = {"params": {"a": 2}}

    merge_yaml(common, model)

    assert common == {"params": {"a": 1}}
    assert model == {"params": {"a": 2}}
