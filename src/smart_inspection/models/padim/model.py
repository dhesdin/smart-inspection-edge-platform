from torch import Tensor, nn
import torch
from torch.utils.data import DataLoader

import torchvision.models as models


from smart_inspection.models.base import AnomalyMethod
from smart_inspection.config.loader import merge_yaml, read_yaml, resolve_config_paths


class PaDiM(AnomalyMethod):
    def __init__(self):
        super().__init__()
        # get conf and merge
        common_yaml_conf = resolve_config_paths(config_path="common.yaml")
        padim_yaml_conf = read_yaml(config_path="padim.yaml")
        merge_yaml_dict = merge_yaml(
            common_config=common_yaml_conf, model_config=padim_yaml_conf
        )
        params_common = merge_yaml_dict["params"]
        backbone = params_common["backbone"]
        layers = params_common["layers"]

        # load and config resnet
        self.resnet = getattr(models, backbone)(weights="DEFAULT")
        for param in self.resnet.parameters():
            param.requires_grad = False

        # hook
        self.features = {}

        def make_hook(layer_name: str) -> None:
            """
            Create a forward hook for the specified layer to capture its output features.
            Args:
                layer_name (str): The name of the layer to hook.
            Returns:
                A hook function that captures the output features of the specified layer.
            """

            def hook(
                module: torch.nn.Module, input: tuple[Tensor], output: Tensor
            ) -> None:
                """
                Hook function to capture the output features of the specified layer.
                Args:
                    module (torch.nn.Module): The module being hooked.
                    input (tuple[Tensor]): The input to the module.
                    output (Tensor): The output of the module.
                """
                self.features[layer_name] = output

            return hook

        # loop from conf and register hook for each layer
        for layer_name in layers:
            layer = getattr(self.resnet, layer_name)
            layer.register_forward_hook(make_hook(layer_name=layer_name))

    def fit(self, train_loader: DataLoader) -> None:
        """
        Fit the model using the provided training data loader.
        Args:
            train_loader (DataLoader): The training data loader.
        """
        pass
    def predict(self, image: Tensor) -> tuple[float, Tensor]:

        pass
