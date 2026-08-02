from smart_inspection.models.base import AnomalyMethod
from smart_inspection.models.padim.model import PaDiM
from smart_inspection.models.stfpm.model import STFPM

METHODS = {"padim": PaDiM, "stfpm": STFPM}


def create_method(method_name: str) -> AnomalyMethod:
    """
    Factory function to create an instance of the specified anomaly detection method.
    Args:
        method_name (str): The name of the anomaly detection method to create. Available methods are: "padim", "stfpm".
    Returns:
        An instance of the specified anomaly detection method.
    Raises:
        ValueError: If the specified method name is not available in the factory.
    """
    if method_name in METHODS:
        return METHODS[method_name]()
    else:
        raise ValueError(f"Invalid method name: {method_name}. Available methods are: {', '.join(METHODS.keys())}")
