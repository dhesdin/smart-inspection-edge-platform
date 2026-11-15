import pytest
import torch
from torch.utils.data import DataLoader

from smart_inspection.models.stfpm.model import STFPM


def test_normalize_features():
    """
    Test the _normalize_features static method of the STFPM class.
    """
    features = torch.tensor([[[[3]], [[4]]]], dtype=torch.float32)  # (B, C, H, W) simplified as (1, 2, 1, 1) for testing

    result = STFPM._normalize_features(features)

    expected = torch.tensor([[[[0.6]], [[0.8]]]])
    assert torch.allclose(result, expected)


def test_normalize_features_zero():
    """
    Test the _normalize_features static method of the STFPM class with a zero feature tensor.
    """
    features = torch.tensor([[[[0]], [[0]]]], dtype=torch.float32)  # (B, C, H, W) simplified as (1, 2, 1, 1) for testing

    result = STFPM._normalize_features(features)

    expected = torch.tensor([[[[0]], [[0]]]], dtype=torch.float32)
    assert torch.allclose(result, expected)


def test_normalize_features_3_channels():
    """
    Test the _normalize_features static method of the STFPM class.
    """
    features = torch.tensor([[[[1]], [[2]], [[2]]]], dtype=torch.float32)  # (B, C, H, W) simplified as (1, 3, 1, 1) for testing

    result = STFPM._normalize_features(features)
    expected = torch.tensor([[[[0.3333]], [[0.6667]], [[0.6667]]]])
    assert torch.allclose(result, expected, atol=1e-3)


def test_compute_distillation_loss():
    """
    Test the _compute_distillation_loss static method of the STFPM class.
    """
    teacher_features = torch.tensor(
        [[[[0.6]], [[0.8]]]],
        dtype=torch.float32,
    )  # (B, C, H, W) simplified as (1, 2, 1, 1) for testing

    student_features = torch.tensor(
        [[[[0.8]], [[0.6]]]],
        dtype=torch.float32,
    )  # (B, C, H, W) simplified as (1, 2, 1, 1) for testing

    result = STFPM._compute_distillation_loss(teacher_features=teacher_features, student_features=student_features)

    # difference : [-0.2, 0.2]
    # squared norm : 0.2**2 + 0.2**2 = 0.08
    # loss = 0.5 * 0.08 = 0.04
    expected = torch.tensor([[[[0.04]]]], dtype=torch.float32)
    assert torch.allclose(result, expected, atol=1e-3)


def test_compute_distillation_loss_identical_features():
    """
    Test the _compute_distillation_loss static method of the STFPM class with identical teacher and student features.
    The expected loss should be zero.
    """
    teacher_features = torch.tensor(
        [[[[0.6]], [[0.8]]]],
        dtype=torch.float32,
    )

    student_features = torch.tensor(
        [[[[0.6]], [[0.8]]]],
        dtype=torch.float32,
    )

    result = STFPM._compute_distillation_loss(
        teacher_features,
        student_features,
    )

    expected = torch.tensor([[[[0.0]]]], dtype=torch.float32)

    assert torch.allclose(result, expected)


def test_compute_spatial_mean_loss():
    """
    Test the _compute_spatial_mean_loss static method of the STFPM class.
    The spatial mean is computed over the height and width dimensions.
    """
    # (B, 1, H, W) = (2, 1, 4, 4): first sample holds 1..16, second holds 17..32
    loss = torch.arange(1, 33, dtype=torch.float32).reshape(2, 1, 4, 4)

    result = STFPM._compute_spatial_mean_loss(loss)

    # mean(1..16) = 136 / 16 = 8.5
    # mean(17..32) = 392 / 16 = 24.5
    expected = torch.tensor([[8.5], [24.5]], dtype=torch.float32)

    assert result.shape == (2, 1)
    assert torch.allclose(result, expected)


def _image_loader(n_images: int, seed: int) -> DataLoader:
    """Small loader of random 64x64 RGB images, in the {"image": tensor} format of the dataset."""
    generator = torch.Generator().manual_seed(seed)
    samples = [{"image": torch.rand(3, 64, 64, generator=generator)} for _ in range(n_images)]
    return DataLoader(samples, batch_size=2)


# get_state -> torch.save -> torch.load(weights_only=True) -> load_state on a new instance gives the same score.
@pytest.mark.integration
def test_state_round_trip_gives_same_score(tmp_path):
    original = STFPM()
    original.param_epochs = 1  # keep the test short: one epoch is enough to move away from the initial weights
    original.fit(_image_loader(n_images=4, seed=0), _image_loader(n_images=2, seed=1))
    image = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(2))
    expected_score, expected_map = original.predict(image)

    state = original.get_state()
    assert all(tensor.device.type == "cpu" for tensor in state.values())
    state_path = tmp_path / "stfpm_state.pt"
    torch.save(state, state_path)

    restored = STFPM()
    restored.load_state(torch.load(state_path, weights_only=True))
    assert not restored.student.training
    score, anomaly_map = restored.predict(image)

    assert torch.allclose(torch.tensor(score), torch.tensor(expected_score))
    assert torch.allclose(anomaly_map, expected_map)


def test_load_state_missing_key_raises():
    # bypass __init__ (config + pretrained weights download): a tiny student is enough to check the keys
    stfpm = STFPM.__new__(STFPM)
    stfpm.device = torch.device("cpu")
    stfpm.student = torch.nn.Linear(2, 2)
    state = {"weight": torch.zeros(2, 2)}  # no "bias"

    with pytest.raises(ValueError, match="bias"):
        stfpm.load_state(state)
