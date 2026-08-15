import pytest

from smart_inspection.models.factory import METHODS, create_method
from smart_inspection.models.padim.model import PaDiM
from smart_inspection.models.stfpm.model import STFPM


def test_create_method_padim_returns_padim_instance():
    """create_method("padim") should return a PaDiM instance."""
    method = create_method("padim")
    assert isinstance(method, PaDiM)


# TODO: This test only checks a basic instantiation as long as STFPM
# is a stub (fit/predict = pass). To be strengthened with real assertions
# on the behavior (fit modifies the internal state, predict returns a
# tuple (float, Tensor) consistently) once the real implementation is done.
def test_create_method_stfpm_returns_stfpm_instance():
    """create_method("stfpm") should return an STFPM instance."""
    method = create_method("stfpm")
    assert isinstance(method, STFPM)


def test_create_method_invalid_name_raises_value_error():
    """create_method should raise ValueError for an unknown method name."""
    with pytest.raises(ValueError):
        create_method("methode_inexistante")


def test_methods_dict_has_expected_keys():
    """METHODS should expose exactly the "padim" and "stfpm" keys."""
    assert set(METHODS.keys()) == {"padim", "stfpm"}
