import json

import numpy as np
import pytest

from smart_inspection.io.save_json import save_json


# The saved file contains exactly the dictionary that was given.
def test_save_json_writes_the_dictionary(tmp_path):
    data = {"a": 1, "b": [1, 2, 3], "c": {"nested": "x"}}
    file_path = tmp_path / "out.json"

    save_json(data, file_path)

    assert json.loads(file_path.read_text()) == data


# A plain string path is accepted like a Path.
def test_save_json_accepts_string_path(tmp_path):
    file_path = tmp_path / "out.json"

    save_json({"a": 1}, str(file_path))

    assert file_path.exists()


# Missing parent directories are created.
def test_save_json_creates_missing_parent_directories(tmp_path):
    file_path = tmp_path / "reports" / "nested" / "out.json"

    save_json({"a": 1}, file_path)

    assert file_path.exists()


# An empty dictionary is rejected and no file is written.
def test_save_json_rejects_empty_dictionary(tmp_path):
    file_path = tmp_path / "out.json"

    with pytest.raises(ValueError, match="empty"):
        save_json({}, file_path)

    assert not file_path.exists()


# An existing file is overwritten, not appended to.
def test_save_json_overwrites_existing_file(tmp_path):
    file_path = tmp_path / "out.json"
    save_json({"a": 1}, file_path)

    save_json({"b": 2}, file_path)

    assert json.loads(file_path.read_text()) == {"b": 2}


# numpy floats (returned by benchmark()) can be serialized.
def test_save_json_serializes_numpy_floats(tmp_path):
    file_path = tmp_path / "out.json"

    save_json({"fps": np.float64(42.5)}, file_path)

    assert json.loads(file_path.read_text()) == {"fps": 42.5}
