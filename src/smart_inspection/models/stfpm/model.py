import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import models

from smart_inspection.config.loader import merge_yaml, read_yaml, resolve_config_paths
from smart_inspection.models.base import AnomalyMethod


class STFPM(AnomalyMethod):
    # https://arxiv.org/pdf/2103.04257

    def __init__(self):
        super().__init__()

        # get conf and merge
        common_yaml_conf = resolve_config_paths(config_path="common.yaml")
        stfpm_yaml_conf = read_yaml(config_path="stfpm.yaml")
        merge_yaml_dict = merge_yaml(common_config=common_yaml_conf, model_config=stfpm_yaml_conf)

        # get params
        params_common = merge_yaml_dict["params"]
        param_backbone = params_common["backbone"]
        _param_epochs = params_common["n_epochs"]
        _param_batch_size = params_common["batch_size"]
        _param_optimizer = params_common["optimizer"]
        _param_momentum = params_common["momentum"]
        _param_learning_rate = params_common["learning_rate"]
        _param_pre_trained = params_common["pre_trained"]
        _param_validation_ratio = params_common["validation_ratio"]
        param_seed = params_common["seed"]
        param_cudnn_deterministic = params_common["cudnn_deterministic"]
        self.device = torch.device(params_common["device"] if torch.cuda.is_available() else "cpu")
        self.layers = params_common["layers"]

        # reproducibility
        torch.manual_seed(param_seed)
        cudnn.deterministic = param_cudnn_deterministic

        # Teacher and Student models
        backbone = getattr(models, param_backbone)
        self.teacher = backbone(weights="DEFAULT")
        self.student = backbone(weights=None)

        # move models to device
        for model in [self.teacher, self.student]:
            model.to(self.device)

        # freeze Teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        # trainable Student
        for p in self.student.parameters():
            p.requires_grad = True

        # Mode only for teacher who stay at eval mode continuously, for student it will be rewrite by fit() and predict()
        self.teacher.eval()

        # hook
        self.student_features = {}
        self.teacher_features = {}

        def make_hook(features_dict: dict, layer_name: str) -> None:
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
                features_dict[layer_name] = output

            return hook

        # loop from conf and register hook for each layer
        for layer_name in self.layers:
            student_layer = getattr(self.student, layer_name)
            teacher_layer = getattr(self.teacher, layer_name)

            student_layer.register_forward_hook(make_hook(features_dict=self.student_features, layer_name=layer_name))
            teacher_layer.register_forward_hook(make_hook(features_dict=self.teacher_features, layer_name=layer_name))

    def fit(self, train_loader: DataLoader) -> None:
        """
        Fit the model using the provided training data loader.
        Args:
            train_loader (DataLoader): The training data loader.
        """
        # https://arxiv.org/pdf/2103.04257 --> Eq(3): ℓ(I_k) = Σ_{l=1}^L α_l × ℓ^l(I_k), with α_l ≥ 0
        pass

    def predict(self, image: Tensor) -> tuple[float, Tensor]:

        pass

    @staticmethod
    def _normalize_features(features: Tensor) -> Tensor:
        """
        Normalize the input features along the channel dimension using the L2 norm.
        Formula: features / sqrt(sum(features**2)).
        Args:
            features (Tensor): The input feature tensor of shape (B, C, H, W).

        Returns:
            Tensor: The normalized feature tensor of the same shape as the input.
        """
        return F.normalize(input=features, p=2, dim=1, eps=1e-12)

    @staticmethod
    def _compute_distillation_loss(teacher_features: Tensor, student_features: Tensor) -> Tensor:
        """
        Compute the distillation loss between teacher and student features.
        The loss is computed as 0.5 * sum((teacher_features - student_features)^2) along the channel dimension.
        Args:
            teacher_features (Tensor): The feature tensor from the teacher model of shape (B, C, H, W).
            student_features (Tensor): The feature tensor from the student model of shape (B, C, H, W).

        Returns:
            Tensor: The computed distillation loss of shape (B, 1, H, W).
        """
        # https://arxiv.org/pdf/2103.04257 --> Eq(1) : l (Ik)i j = 1 2 Fˆ l t (Ik)i j −Fˆ l s (Ik)i j 2 2 ,
        difference = teacher_features - student_features  # (B,C,H,W)

        # norm = torch.sqrt(torch.sum(input=difference**2, dim=1, keepdim=True))
        # --> norm² = (sqrt(sum sq))² -> sqrt cancels with the **2, so just sum of squares
        loss = 0.5 * torch.sum(input=difference**2, dim=1, keepdim=True)
        return loss

    @staticmethod
    def _compute_spatial_mean_loss(loss: Tensor) -> Tensor:
        """
        Compute the spatial mean of the input loss tensor.
        The mean is computed along the height and width dimensions.
        Args:
            loss (Tensor): The input loss tensor of shape (B, 1, H, W).

        Returns:
            Tensor: The spatial mean loss tensor of shape (B,1).
        """
        # https://arxiv.org/pdf/2103.04257 --> Eq(2): ℓ^l(I_k) = (1 / (w_l × h_l)) × Σ_i Σ_j ℓ^l(I_k)_{ij}
        return loss.mean(dim=(2, 3))
