import numpy as np


# matched-filter algorithm
def matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    iterate: bool,
    albedoadjust: bool,
) -> np.ndarray:
    """Calculate the methane enhancement of the image data based on the original matched filter method.

    Args:
        data_cube (np.ndarray): 3D array representing the image data cube.
        unit_absorption_spectrum (np.ndarray): 1D array representing the unit absorption spectrum.
        iterate (bool): Flag indicating whether to perform iterative computation.
        albedoadjust (bool): Flag indicating whether to adjust for albedo.

    Returns:
        np.ndarray: 2D array representing the concentration of methane.
    """
    # Get dimensions
    _, rows, cols = data_cube.shape
    concentration = np.zeros((rows, cols))

    # Calculate background spectrum and target spectrum ignoring NaN values
    background_spectrum = np.nanmean(data_cube, axis=(1, 2))
    target_spectrum = background_spectrum * unit_absorption_spectrum

    # Calculate radiance difference while handling NaNs
    radiancediff_with_background = data_cube - background_spectrum[:, None, None]
    radiancediff_with_background = np.nan_to_num(radiancediff_with_background, nan=0.0)

    # Compute covariance and its inverse
    d_covariance = radiancediff_with_background
    covariance = np.tensordot(d_covariance, d_covariance, axes=((1, 2), (1, 2))) / (
        rows * cols
    )
    covariance_inverse = np.linalg.pinv(covariance)

    # Adjust albedo if needed, ignoring NaNs
    albedo = np.ones((rows, cols))
    if albedoadjust:
        albedo = np.einsum("ijk,i->jk", data_cube, background_spectrum) / np.dot(
            background_spectrum, background_spectrum
        )
        albedo = np.nan_to_num(albedo, nan=1.0)

    # Pre-compute common denominator
    common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum

    # Vectorized concentration computation
    numerator = np.einsum(
        "ijk,i->jk",
        radiancediff_with_background,
        np.dot(covariance_inverse, target_spectrum),
    )
    concentration = numerator / (albedo * common_denominator)

    # Handle iteration for more accurate concentration calculation
    if iterate:
        for _ in range(5):
            # Update background and target spectra
            residual = (
                data_cube
                - (albedo * concentration)[None, :, :] * target_spectrum[:, None, None]
            )
            background_spectrum = np.nanmean(residual, axis=(1, 2))
            target_spectrum = background_spectrum * unit_absorption_spectrum

            # Update radiance difference with new background
            radiancediff_with_background = (
                data_cube - background_spectrum[:, None, None]
            )
            radiancediff_with_background = np.nan_to_num(
                radiancediff_with_background, nan=0.0
            )

            # Update covariance
            d_covariance = (
                radiancediff_with_background
                - (albedo * concentration)[None, :, :] * target_spectrum[:, None, None]
            )
            covariance = np.tensordot(
                d_covariance, d_covariance, axes=((1, 2), (1, 2))
            ) / (rows * cols)
            covariance_inverse = np.linalg.pinv(covariance)

            # Update common denominator
            common_denominator = (
                target_spectrum.T @ covariance_inverse @ target_spectrum
            )

            # Recompute concentration vectorized
            numerator = np.einsum(
                "ijk,i->jk",
                radiancediff_with_background,
                np.dot(covariance_inverse, target_spectrum),
            )
            concentration = np.maximum(numerator / (albedo * common_denominator), 0)

    return concentration


def matched_filter_test():
    test_image = np.random.rand(10, 10, 10)
    unit_absorption_spectrum = np.random.rand(10)
    iterate = True
    albedoadjust = True
    result = matched_filter(test_image, unit_absorption_spectrum, iterate, albedoadjust)
    print(result)


if __name__ == "__main__":
    
