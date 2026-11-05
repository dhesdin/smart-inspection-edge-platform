import torch


def split_data(
    n_images: int,
    calibration_ratio: float,
    validation_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split image indices into calibration, validation and training sets.

    The validation ratio is applied to the images remaining
    after the calibration split. Sizes are rounded down with int().
    All parameters are required: values must come from the config.

    Args:
        n_images (int): Total number of images to split.
        calibration_ratio (float): Fraction of all images used for calibration, in [0, 1].
        validation_ratio (float): Fraction of the non-calibration images used for validation, in [0, 1].
        seed (int): Seed of the random permutation.

    Returns:
        tuple[list[int], list[int], list[int]]: Calibration, validation and training indices.

    Raises:
        ValueError: If n_images <= 0, if a ratio is outside [0, 1],
            or if the resulting training set is empty.
    """
    # guard clauses
    if n_images <= 0:
        raise ValueError("n_images must be greater than 0.")

    if not 0 <= calibration_ratio <= 1:
        raise ValueError("calibration_ratio must be between 0 and 1.")

    if not 0 <= validation_ratio <= 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    # reproducible random permutation of image indices

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_images, generator=generator).tolist()

    # Calculate split sizes
    n_calibration = int(n_images * calibration_ratio)
    n_remaining = n_images - n_calibration
    n_validation = int(n_remaining * validation_ratio)

    if n_images - n_calibration - n_validation <= 0:
        raise ValueError("Ratios leave no image for the training set.")

    # Split indices
    calibration_indices = indices[:n_calibration]  # First n_calibration indices for calibration set

    validation_indices = indices[n_calibration : n_calibration + n_validation]  # Next n_validation indices for validation set

    training_indices = indices[n_calibration + n_validation :]  # Remaining indices for training set

    return calibration_indices, validation_indices, training_indices
