import torch

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
