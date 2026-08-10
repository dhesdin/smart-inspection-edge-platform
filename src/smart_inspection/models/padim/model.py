import torch
import torchvision.models as models
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from smart_inspection.config.loader import merge_yaml, read_yaml, resolve_config_paths
from smart_inspection.models.base import AnomalyMethod


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
        self.layers = params_common["layers"]
        self.device = torch.device(
            params_common["device"] if torch.cuda.is_available() else "cpu"
        )

        # load resnet
        self.resnet = getattr(models, backbone)(weights="DEFAULT")

        # device get attr
        self.resnet.to(self.device)
        # freeze

        for param in self.resnet.parameters():
            param.requires_grad = False

        # eval
        self.resnet.eval()

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
        for layer_name in self.layers:
            layer = getattr(self.resnet, layer_name)
            layer.register_forward_hook(make_hook(layer_name=layer_name))

    def fit(self, train_loader: DataLoader) -> None:
        """
        Fit the model using the provided training data loader.
        Args:
            train_loader (DataLoader): The training data loader.
        """
        # === First step : accumulate the tensors of the features for each layer in a dictionary of lists ===
        features_accumulator = {layer_name: [] for layer_name in self.layers}
        # forward pass through the training data to collect features
        for batch in train_loader:
            images = batch["image"].to(self.device)
            self.resnet(images)  # forward
            for layer_name, feature in self.features.items():
                features_accumulator[layer_name].append(feature)
        # === Second step : convert the list of features tensors to single one  ===
        for layer_name, features_list in features_accumulator.items():
            features_accumulator[layer_name] = torch.cat(tensors=features_list, dim=0)

        # === Thir step : upsampling all layers ===
        target_size = features_accumulator[self.layers[0]].shape[2:]

        for layer_name in self.layers[1:]:
            layer_upsampled = F.interpolate(
                features_accumulator[layer_name], size=target_size, mode="bilinear"
            )
            features_accumulator[layer_name] = layer_upsampled
        # === Fourth step : concat layers into a single layer for channels ===
        feature_concat = torch.cat(
            tensors=[features_accumulator[layer_name] for layer_name in self.layers],
            dim=1,
        )
        self.embeddings = feature_concat
        # === fifth step :permute and reshape from (N,C,H,W) to (H*W, N, C) (2D -> 1D - flat) ===
        self.embeddings = self.embeddings.permute(2, 3, 0, 1)
        self.embeddings = self.embeddings.reshape(
            self.embeddings.shape[0] * self.embeddings.shape[1],
            self.embeddings.shape[2],
            self.embeddings.shape[3],
        )

        # === sixth step :Mean and cov ===
        self.mean = torch.mean(self.embeddings, dim=1)  # (HW,C)

        # cov --> https://arxiv.org/abs/2011.08785 --> Σij = 1 N − 1 X N k=1 (x k ij − µij)(x k ij − µij) T + eI
        mean = self.mean.unsqueeze(
            dim=1
        )  # (HW,1,C) for substraction between embeddings and mean
        centered = self.embeddings - mean  # (HW,N,C)

        transposed = centered.transpose(1, 2)  # (HW,C,N)
        cov = torch.matmul(transposed, centered)  # (C,N) @ (N,C) -> (C,C)
        self.cov = cov / (self.embeddings.shape[1] - 1)

    def predict(self, image: Tensor) -> tuple[float, Tensor]:

        pass
