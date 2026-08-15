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


# TODO --> regularization test +εI.
# TODO --> mahalanobis test