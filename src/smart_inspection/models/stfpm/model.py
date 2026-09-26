import copy
from pathlib import Path

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
        self.param_input_size = (
            params_common["input_size"],
            params_common["input_size"],
        )
        self.param_epochs = params_common["n_epochs"]
        self.param_optimizer = params_common["optimizer"]
        self.param_momentum = params_common["momentum"]
        self.param_learning_rate = params_common["learning_rate"]
        self.param_validation_split_ratio = params_common["validation_split_ratio"]
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

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        """
        Fit the model using the provided training and validation data loaders.
        TW : Requires_grad and device are in __init__
        Args:
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader | None, optional): The validation data loader.
                Required by any method that needs validation ; used for best-model
                selection across epochs.
        Raises:
            ValueError: If the validation data loader is not provided.
        Returns:
            None
        """

        if val_loader is None:
            raise ValueError("STFPM.fit() requires a validation data loader.")
        # Set the student model to training mode and initialize the optimizer
        optimizer_class = getattr(torch.optim, self.param_optimizer.upper())
        optimizer = optimizer_class(
            params=self.student.parameters(),
            lr=self.param_learning_rate,
            momentum=self.param_momentum,
        )

        # best val to compare and save the best model during training
        best_val_loss = None
        best_student_state = None

        for epoch in range(self.param_epochs):
            self.student.train()
            train_losses = []
            for batch in train_loader:
                # ====== TRAIN ====== #
                # Load the input image from the batch
                image = batch["image"].to(device=self.device)
                optimizer.zero_grad()
                # Forward pass through the teacher and student models
                with torch.no_grad():
                    self.teacher(image)
                self.student(image)

                total_loss = self._compute_total_loss()
                # backward and step for student
                total_loss.backward()
                optimizer.step()

                train_losses.append(total_loss.item())
            mean_train_loss = sum(train_losses) / len(train_losses)

            # ====== EVAL ====== #
            val_losses = []
            self.student.eval()

            with torch.no_grad():
                for batch in val_loader:
                    image = batch["image"].to(device=self.device)

                    self.teacher(image)
                    self.student(image)

                    total_loss = self._compute_total_loss()
                    val_losses.append(total_loss.item())

                mean_val_losses = sum(val_losses) / len(val_losses)
            # ====== PRINT ====== #
            print(
                f"Epoch {epoch + 1}/{self.param_epochs}: Mean training loss: {mean_train_loss:.6f}"
                + f" | Mean validation loss: {mean_val_losses:.6f}"
            )

            # ====== SAVE BEST MODEL ====== #
            if best_val_loss is None or mean_val_losses < best_val_loss:
                best_val_loss = mean_val_losses
                best_student_state = copy.deepcopy(self.student.state_dict())
                print(f"Epoch {epoch + 1}: New best validation loss: {best_val_loss:.6f}")

        # ====== RESTORE BEST MODEL ====== #
        if best_student_state is None:
            raise ValueError("No best student state found. Training might not have been performed.")
        self.student.load_state_dict(best_student_state)

    def predict(self, image: Tensor) -> tuple[float, Tensor]:
        # 3.3 Section of papers
        """
        Predict the anomaly score and anomaly map for a given input image.

        Args:
            image (Tensor): The input image tensor of shape (C, H, W).

        Returns:
            tuple[float, Tensor]: A tuple containing the anomaly score (float) and the anomaly map (Tensor).
        """
        # forward teacher/student but w/o optimizer backward
        self.student.eval()
        image = image.unsqueeze(dim=0).to(device=self.device)
        target_size = self.param_input_size
        with torch.no_grad():
            self.teacher(image)
            self.student(image)
        # a loop on self layers for compute
        layer_losses = []
        for layer_name in self.layers:
            student_features = self._normalize_features(self.student_features[layer_name])
            teacher_features = self._normalize_features(self.teacher_features[layer_name])
            loss_dist = self._compute_distillation_loss(teacher_features=teacher_features, student_features=student_features)

            # Upsampling each layer toward the size on input image
            loss_dist = F.interpolate(input=loss_dist, size=target_size, mode="bilinear")

            layer_losses.append(loss_dist)

        # product (element-wise) of the 3 upsampled feature maps
        anomaly_map = layer_losses[0]
        for loss_map in layer_losses[1:]:
            anomaly_map *= loss_map

        # from (B,1,H,W to (H,W))
        anomaly_map = anomaly_map.squeeze()
        # Papers -> maximum value of the element-wise product across the spatial dimensions.
        score = anomaly_map.max().item()

        return score, anomaly_map

    @property
    def validation_split_ratio(self) -> float | None:
        return self.param_validation_split_ratio

    def save(self, file_path: Path) -> None:
        """
        Save the model to the specified file path.
        Args:
            file_path (Path): The path to save the model.
        """
        torch.save(self.student.state_dict(), file_path)

    def load(self, file_path: Path) -> None:
        """
        Load the model from the specified file path.
        Args:
            file_path (Path): The path to load the model from.
        """
        self.student.load_state_dict(torch.load(file_path, map_location=self.device))

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

    def _compute_total_loss(self) -> Tensor:
        """Compute the total STFPM distillation loss."""
        # Compute the distillation loss for each layer
        layer_losses = []
        for layer_name in self.layers:
            teacher_features = self.teacher_features[layer_name]
            student_features = self.student_features[layer_name]

            teacher_features = self._normalize_features(teacher_features)
            student_features = self._normalize_features(student_features)

            loss = self._compute_distillation_loss(teacher_features=teacher_features, student_features=student_features)
            mean_loss = self._compute_spatial_mean_loss(loss=loss)
            layer_losses.append(mean_loss)
            # https://arxiv.org/pdf/2103.04257 --> Eq(3): ℓ(I_k) = Σ_{l=1}^L α_l × ℓ^l(I_k), with α_l ≥ 0
            # α_l in papers is equal to 1

        total_loss = sum(layer_losses)  # sum image[layer] : img1[layer1] + img1[layer2] + ... : shape (B,1)
        total_loss = total_loss.mean()  # mean over the batch dim : shape (1)
        return total_loss
