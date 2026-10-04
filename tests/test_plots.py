import matplotlib

matplotlib.use("Agg")  # no display needed, also in CI

import matplotlib.pyplot as plt
import pytest
import torch

from smart_inspection.visualization.plots import _denormalize_image, generate_comparison_plot, generate_plots

SIZE = 8


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _images(n: int) -> list[torch.Tensor]:
    return [torch.zeros(3, SIZE, SIZE) for _ in range(n)]


def _masks(n: int) -> list[torch.Tensor]:
    return [torch.zeros(1, SIZE, SIZE) for _ in range(n)]


def _maps(n: int) -> list[torch.Tensor]:
    return [torch.rand(SIZE, SIZE) for _ in range(n)]


# A (3, H, W) image comes back as (H, W, 3), the layout imshow expects.
def test_denormalize_image_returns_hwc() -> None:
    assert _denormalize_image(torch.zeros(3, SIZE, SIZE)).shape == (SIZE, SIZE, 3)


# An image that is not (3, H, W) is rejected.
@pytest.mark.parametrize("shape", [(SIZE, SIZE), (1, SIZE, SIZE), (1, 3, SIZE, SIZE)])
def test_denormalize_image_rejects_wrong_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        _denormalize_image(torch.zeros(shape))


# The single-method figure has the image, the mask and the anomaly map.
def test_generate_plots_has_three_axes() -> None:
    fig = generate_plots(image=_images(1)[0], mask=_masks(1)[0], anomaly_map=_maps(1)[0], vmin=0.0, vmax=1.0)

    assert [ax.get_title() for ax in fig.axes] == ["Image", "Ground truth mask", "Anomaly map"]


# One row per image with 4 columns (image, mask, padim, stfpm), also with a single image (squeeze=False).
@pytest.mark.parametrize("num_images", [1, 3])
def test_comparison_plot_layout(num_images: int) -> None:
    fig = generate_comparison_plot(
        images=_images(num_images),
        masks=_masks(num_images),
        anomaly_maps={"padim": _maps(num_images), "stfpm": _maps(num_images)},
        vmax={"padim": 30.0, "stfpm": 1.0},
    )

    # 4 columns (image, mask, padim, stfpm) per row
    assert len(fig.axes) == 4 * num_images
    assert [ax.get_title() for ax in fig.axes[:4]] == ["Image", "Ground truth mask", "PADIM", "STFPM"]


# Each method keeps its own color scale on every row.
def test_comparison_plot_uses_one_scale_per_method() -> None:
    fig = generate_comparison_plot(
        images=_images(2),
        masks=_masks(2),
        anomaly_maps={"padim": _maps(2), "stfpm": _maps(2)},
        vmax={"padim": 30.0, "stfpm": 1.0},
    )

    for row in range(2):
        padim_ax, stfpm_ax = fig.axes[4 * row + 2], fig.axes[4 * row + 3]
        assert padim_ax.images[0].get_clim() == (0.0, 30.0)
        assert stfpm_ax.images[0].get_clim() == (0.0, 1.0)


# Fewer masks than images is rejected.
def test_comparison_plot_rejects_wrong_number_of_masks() -> None:
    with pytest.raises(ValueError, match="masks"):
        generate_comparison_plot(
            images=_images(2),
            masks=_masks(1),
            anomaly_maps={"padim": _maps(2)},
            vmax={"padim": 1.0},
        )


# A method with fewer anomaly maps than images is rejected.
def test_comparison_plot_rejects_wrong_number_of_maps() -> None:
    with pytest.raises(ValueError, match="stfpm"):
        generate_comparison_plot(
            images=_images(2),
            masks=_masks(2),
            anomaly_maps={"padim": _maps(2), "stfpm": _maps(1)},
            vmax={"padim": 1.0, "stfpm": 1.0},
        )


# A method without a vmax is rejected.
def test_comparison_plot_rejects_missing_vmax() -> None:
    with pytest.raises(ValueError, match="vmax"):
        generate_comparison_plot(
            images=_images(2),
            masks=_masks(2),
            anomaly_maps={"padim": _maps(2), "stfpm": _maps(2)},
            vmax={"padim": 1.0},
        )
