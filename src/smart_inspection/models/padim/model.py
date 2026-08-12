import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models
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
        merge_yaml_dict = merge_yaml(common_config=common_yaml_conf, model_config=padim_yaml_conf)

        # get params
        params_common = merge_yaml_dict["params"]
        param_backbone = params_common["backbone"]
        self.param_n_features = params_common["n_features"]
        param_seed = params_common["seed"]
        param_cudnn_deterministic = params_common["cudnn_deterministic"]

        # reproducibility
        torch.manual_seed(param_seed)
        cudnn.deterministic = param_cudnn_deterministic

        self.layers = params_common["layers"]
        self.device = torch.device(params_common["device"] if torch.cuda.is_available() else "cpu")
        # load resnet
        self.resnet = getattr(models, param_backbone)(weights="DEFAULT")

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

            def hook(module: torch.nn.Module, input: tuple[Tensor], output: Tensor) -> None:
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

        with torch.no_grad():  # disable gradient computation for efficiency
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
            layer_upsampled = F.interpolate(features_accumulator[layer_name], size=target_size, mode="bilinear")
            # === Fourth step : concat layers into a single layer for channels ===
            features_accumulator[layer_name] = layer_upsampled
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
        # rand and keep only 100 first one
        self.selected_indices = torch.randperm(self.embeddings.shape[2])[: self.param_n_features]
        self.embeddings = self.embeddings[:, :, self.selected_indices]  # where C = param_n_features
        self.mean = torch.mean(self.embeddings, dim=1)  # (HW,C)

        # cov --> https://arxiv.org/abs/2011.08785 --> Σij = 1 N − 1 X N k=1 (x k ij − µij)(x k ij − µij) T + eI
        mean = self.mean.unsqueeze(dim=1)  # (HW,1,C) for substraction between embeddings and mean
        centered = self.embeddings - mean  # (HW,N,C)

        transposed = centered.transpose(1, 2)  # (HW,C,N)
        cov = torch.matmul(transposed, centered)  # (C,N) @ (N,C) -> (C,C)

        # identity matrix for avoid singular matrix and loop [i][i]
        epsilon = 1e-2
        identity_m = torch.eye(n=self.param_n_features, device=self.device)
        self.cov = cov / (self.embeddings.shape[1] - 1)
        self.cov = self.cov + (epsilon * identity_m)

    def predict(self, image: Tensor) -> tuple[float, Tensor]:
        image = image.unsqueeze(0).to(device=self.device)  # add batch for forward

        with torch.no_grad():
            self.resnet(image)

        # === Upsampling ===
        target_size = self.features[self.layers[0]].shape[2:]
        for layer_name in self.layers[1:]:
            layer_upsampled = F.interpolate(self.features[layer_name], size=target_size, mode="bilinear")
            self.features[layer_name] = layer_upsampled
        # === concat layers ===
        features_concat = torch.cat(tensors=[self.features[layer_name] for layer_name in self.features], dim=1)
        # === Permute and reshape ===
        embeddings = features_concat  # (N,C,H,W)
        embeddings = embeddings.permute(2, 3, 0, 1)  # (H,W,N,C)
        embeddings = embeddings.reshape(embeddings.shape[0] * embeddings.shape[1], embeddings.shape[2] * embeddings.shape[3])  # (HW,C) (64*64, 1*448)

        # === reduce channels ===
        embeddings = embeddings[:, self.selected_indices]
        # === Mahalanobis distance ===

        # TODO
