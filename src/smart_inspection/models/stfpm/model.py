import torch
import torch.backends.cudnn as cudnn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import models

from smart_inspection.config.loader import merge_yaml, read_yaml, resolve_config_paths
from smart_inspection.models.base import AnomalyMethod


class STFPM(AnomalyMethod):
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

    def fit(self, train_loader: DataLoader) -> None:
        """
        Fit the model using the provided training data loader.
        Args:
            train_loader (DataLoader): The training data loader.
        """
        pass

    def predict(self, image: Tensor) -> tuple[float, Tensor]:

        pass
