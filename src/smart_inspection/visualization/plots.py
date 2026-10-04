import matplotlib.pyplot as plt
import torch
from torch import Tensor

from smart_inspection.data.dataset import IMAGENET_MEAN, IMAGENET_STD


def _denormalize_image(image: Tensor) -> Tensor:
    """Denormalize an image tensor using the ImageNet mean and standard deviation.

    Args:
        image (Tensor): The input image tensor of shape (C, H, W).

    Returns:
        Tensor: The denormalized image tensor of shape (H, W, C) suitable for visualization.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected an image of shape (3, H, W), got {tuple(image.shape)}")

    image_mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(image.device)  # (3,1,1)
    image_std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(image.device)  # (3,1,1)
    # normalize is (pixel - mean) / std --> normalize *std + mean
    unnormalized_image = image * image_std + image_mean  # (C,H,W)

    unnormalized_image = unnormalized_image.permute(1, 2, 0)  # (H,W,C) for pltimshow

    return unnormalized_image


def generate_plots(image: Tensor, mask: Tensor, anomaly_map: Tensor, vmin: float | None = None, vmax: float | None = None) -> plt.Figure:
    """Generate plots for an image, its ground truth mask, and the corresponding anomaly map.

    Args:
        image (Tensor): The input image tensor of shape (C, H, W).
        mask (Tensor): The ground truth mask tensor of shape (1, H, W) or (H, W).
        anomaly_map (Tensor): The anomaly map tensor of shape (H, W).
        vmin (float | None, optional): Minimum value for the anomaly map colormap. Defaults to None.
        vmax (float | None, optional): Maximum value for the anomaly map colormap. Defaults to None.

    Returns:
        plt.Figure: The generated matplotlib figure containing the image, mask, and anomaly map.

    """
    unnormalized_image = _denormalize_image(image)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # to CPU and numpy for matplotlib
    unnormalized_image = unnormalized_image.cpu().numpy()
    mask = mask.cpu().numpy()
    anomaly_map = anomaly_map.cpu().numpy()

    # axes 0 : original image
    axes[0].imshow(unnormalized_image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    # axes 1 : ground truth mask
    axes[1].imshow(mask.squeeze(), cmap="gray")
    axes[1].set_title("Ground truth mask")
    axes[1].axis("off")

    # axes 2 : anomaly map
    axes[2].imshow(anomaly_map, cmap="jet", vmin=vmin, vmax=vmax)
    axes[2].set_title("Anomaly map")
    axes[2].axis("off")

    fig.tight_layout()  # adjust subplots to fit into figure area.
    return fig


def generate_comparison_plot(
    images: list[Tensor],
    masks: list[Tensor],
    anomaly_maps: dict[str, list[Tensor]],
    vmax: dict[str, float],
    vmin: float = 0.0,
) -> plt.Figure:
    """Generate a side-by-side comparison of several methods on the same images.

    One row per image, columns: image, ground truth mask, then one anomaly map per method.
    Each method keeps its own color scale, since anomaly scores are not comparable across methods.

    Args:
        images (list[Tensor]): A list of input image tensors of shape (C, H, W).
        masks (list[Tensor]): A list of ground truth mask tensors of shape (1, H, W) or (H, W).
        anomaly_maps (dict[str, list[Tensor]]): Anomaly maps per method, e.g. {"padim": [...], "stfpm": [...]}.
            Each list holds one map of shape (H, W) per image, in the same order as `images`.
        vmax (dict[str, float]): Maximum value of the colormap per method, with the same keys as `anomaly_maps`.
        vmin (float, optional): Minimum value of the colormap, shared by all methods. Defaults to 0.0.

    Returns:
        plt.Figure: The generated matplotlib figure containing the comparison plots.
    """
    num_images = len(images)
    methods = list(anomaly_maps.keys())

    if len(masks) != num_images:
        raise ValueError(f"Expected {num_images} masks, got {len(masks)}")
    for method in methods:
        if len(anomaly_maps[method]) != num_images:
            raise ValueError(f"Expected {num_images} anomaly maps for '{method}', got {len(anomaly_maps[method])}")
        if method not in vmax:
            raise ValueError(f"Missing vmax for method '{method}'")

    num_cols = 2 + len(methods)
    # squeeze=False: axes is always 2D (num_images, num_cols), even with a single image
    fig, axes = plt.subplots(num_images, num_cols, figsize=(5 * num_cols, 5 * num_images), squeeze=False)

    for i in range(num_images):
        # axes 0 : original image
        axes[i][0].imshow(_denormalize_image(images[i]).cpu().numpy())
        axes[i][0].set_title("Image")
        axes[i][0].axis("off")

        # axes 1 : ground truth mask
        axes[i][1].imshow(masks[i].cpu().numpy().squeeze(), cmap="gray")
        axes[i][1].set_title("Ground truth mask")
        axes[i][1].axis("off")

        # axes 2 : one anomaly map per method, each with its own scale
        for j, method in enumerate(methods):
            axes[i][2 + j].imshow(anomaly_maps[method][i].cpu().numpy(), cmap="jet", vmin=vmin, vmax=vmax[method])
            axes[i][2 + j].set_title(method.upper())
            axes[i][2 + j].axis("off")

    fig.tight_layout()
    return fig
