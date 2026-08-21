import torch

from smart_inspection.models.padim.model import PaDiM


# First test bessel/cov
def test_compute_covariance_use_bessel_correction():
    """
    Test the _compute_covariance method of the PaDiM class with Bessel's correction
    to ensure that the covariance is computed correctly.
    """
    # manually test, mean = 2 , gap by mean is [-1,0,1], squaresum = 2, n-1 = 2, var =squaresum / n-1 --> 1

    embeddings = torch.tensor([[[1.0], [2.0], [3.0]]])  # HW,N,C
    mean = torch.tensor([[[2.0]]])

    cov = PaDiM._compute_covariance(embeddings=embeddings, mean=mean)

    expected_cov = torch.tensor([[[1.0]]])
    assert torch.isclose(cov, expected_cov).all()


# Covariance Regularization test
def test_compute_covariance_regularization():
    """
    Test the _regularize_covariance method of the PaDiM class to ensure that the covariance
    matrix is regularized correctly by adding a small value to its diagonal elements.
    """
    # shape(1,2,2)
    cov = torch.tensor([[[1, 2], [2, 4]]], dtype=torch.float32)  # HW,C,C

    regularized_cov = PaDiM._regularize_covariance(cov=cov, epsilon=1e-2)

    # verify results by manual tests on this matrix
    expected_cov = torch.tensor(
        [[[1.01, 2.0], [2.0, 4.01]]], dtype=torch.float32
    )  # HW,C,C
    assert torch.isclose(input=regularized_cov, other=expected_cov).all()


def test_regularized_covariance_inverse():
    """
    Test the inverse of the regularized covariance matrix to ensure that it is computed correctly.
    """
    # shape(1,2,2)
    cov = torch.tensor([[[1, 2], [2, 4]]], dtype=torch.float32)  # HW,C,C

    regularized_cov = PaDiM._regularize_covariance(cov=cov, epsilon=1e-2)

    inv_cov_matrix = torch.linalg.inv(regularized_cov)  # (HW,C,C)

    # verify results by manual tests on this matrix
    expected_inverse = torch.tensor(
        [[[80.0399, -39.92], [-39.92, 20.1596]]], dtype=torch.float32
    )  # HW,C,C
    # rtol×∣other∣+atol so --> 10^-5 * 80.0399 + 10^-3 = 0.001800399 for tolerance
    assert torch.isclose(input=inv_cov_matrix, other=expected_inverse, atol=1e-3).all()


def test_mahalanobis_distance():
    """
    Test the _compute_mahalanobis_distance method of the PaDiM class to ensure that the Mahalanobis distance
    is computed correctly.
    """

    # distance² = (x - µ)^T * Σ⁻¹ * (x - µ)   --> line * inv_mat_cov * column
    # Build Identity matrix for avoid the sigma bc I⁻1 = I --> (x - µ)^T * (x - µ)

    embeddings = torch.tensor([[3, 4]], dtype=torch.float32)  # (HW,C) -> (1,2)
    mean = torch.tensor([[1, 1]], dtype=torch.float32)  # (HW,C) -> (1,2)
    cov = torch.eye(2).unsqueeze(0)  # identity mat  (1,2,2)

    square_dist = PaDiM._compute_mahalanobis_distance(
        embeddings=embeddings, mean=mean, cov=cov
    )

    # x-u = (2,3) --> (x-u) = shape (1,2)
    # (x-u)^T --> shape (2,1) : (1,2) @ (2,1) = (1,1)

    # results : 2x2 + 3x3 = 13 = (square_dist)
    expected_square_dist = torch.tensor([[[13]]], dtype=torch.float32)  # (HW,1,1)

    assert torch.isclose(input=square_dist, other=expected_square_dist).all()
