from torchvision.models import Weights, get_model_weights


def resolve_backbone_weights(backbone: str, backbone_weights: str) -> Weights:
    """
    Resolve the pretrained weights of a torchvision backbone from their names in the config.
    Args:
        backbone (str): The torchvision model name, e.g. "resnet18".
        backbone_weights (str): The weights name, e.g. "IMAGENET1K_V1".
    Returns:
        Weights: The torchvision weights entry.
    Raises:
        ValueError: If backbone_weights is not a weights name of this backbone.
    """
    weights_enum = get_model_weights(backbone)
    try:
        return weights_enum[backbone_weights]
    except KeyError:
        available = [weights.name for weights in weights_enum]
        raise ValueError(f"Unknown weights {backbone_weights!r} for backbone {backbone!r}. Available weights: {available}") from None
