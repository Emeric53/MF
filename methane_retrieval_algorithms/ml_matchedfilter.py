import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import time
import os

from utils import generate_radiance_lut_and_uas as glut
from utils import simulate_images as si
from utils import satellites_data as sd


# def ml_matched_filter(
#     data_cube: np.ndarray,
#     unit_absorption_spectrum: np.ndarray,
#     iterate: bool = False,
#     albedoadjust: bool = False,
#     sparsity: bool = False,
# ) -> np.ndarray:
#     """Calculate the methane enhancement of the image data based on the original matched filter method.

#     Args:
#         data_cube (np.ndarray): 3D array representing the image data cube.
#         unit_absorption_spectrum (np.ndarray): 1D array representing the unit absorption spectrum.
#         iterate (bool): Flag indicating whether to perform iterative computation.
#         albedoadjust (bool): Flag indicating whether to adjust for albedo.

#     Returns:
#         np.ndarray: 2D array representing the concentration of methane.
#     """
#     # Ensure data_cube is a 3D array
#     if data_cube.ndim == 2:
#         data_cube = data_cube[np.newaxis, :, :]
#     concentration = mf()
#     bands, rows, cols = data_cube.shape

#     concentration = mf.matched_filter(
#         data_cube, unit_absorption_spectrum, iterate, albedoadjust, sparsity
#     )

#     std = np.std(concentration)
#     mean = np.mean(concentration)
#     max = np.max(concentration)
#     threshold = mean + std
#     if threshold < 500:
#         threshold = 500
#     bg_uas = glut.generate_satellite_uas_for_specific_range_from_lut(
#         "AHSI", 0, threshold, 2150, 2500, 25, 0
#     )

#     if max > 10000:
#         pass

#     # Step 1: Calculate background spectrum and target spectrum
#     background_spectrum = np.nanmean(
#         data_cube, axis=(1, 2)
#     )  # Mean across rows and cols
#     target_spectrum = background_spectrum * bg_uas
#     background_spectrum += 5000 * target_spectrum
#     target_spectrum = background_spectrum * bg_uas2
#     background_spectrum += 5000 * target_spectrum
#     target_spectrum = background_spectrum * unit_absorption_spectrum

#     # Step 2: Calculate radiance difference (handling NaNs)
#     radiancediff_with_background = data_cube - background_spectrum[:, None, None]

#     # Step 3: Compute covariance matrix (avoid NaNs)
#     d_covariance = radiancediff_with_background - 10000 * target_spectrum[:, None, None]
#     covariance = np.tensordot(d_covariance, d_covariance, axes=((1, 2), (1, 2))) / (
#         rows * cols
#     )
#     covariance += np.eye(covariance.shape[0]) * 1e-6  # Regularization
#     covariance_inverse = np.linalg.inv(covariance)

#     # Step 4: Adjust albedo if needed
#     albedo = np.ones((rows, cols))
#     if albedoadjust:
#         albedo = np.einsum("ijk,i->jk", data_cube, background_spectrum) / np.dot(
#             background_spectrum, background_spectrum
#         )
#         albedo = np.nan_to_num(albedo, nan=1.0)

#     # Step 5: Precompute common denominator
#     common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum

#     # Step 6: Compute concentration (vectorized)
#     numerator = np.einsum(
#         "ijk,i->jk",
#         radiancediff_with_background,
#         np.dot(covariance_inverse, target_spectrum),
#     )
#     concentration = numerator / (albedo * common_denominator) + 11000

#     # Step 7: Handle iteration for more accurate concentration calculation
#     if iterate:
#         l1filter = np.zeros((rows, cols))
#         for _ in range(5):
#             # Step 7.0: Handle sparsity
#             if sparsity:
#                 l1filter = 1 / (concentration + 1e-6)

#             # Step 7.1: Update residual (background and target spectra)
#             residual = (
#                 data_cube
#                 - (albedo * concentration)[None, :, :] * target_spectrum[:, None, None]
#             )

#             background_spectrum = np.nanmean(residual, axis=(1, 2))

#             target_spectrum = background_spectrum * unit_absorption_spectrum

#             # Step 7.2: Recompute radiance difference and covariance
#             radiancediff_with_background = (
#                 data_cube - background_spectrum[:, None, None]
#             )

#             d_covariance = (
#                 radiancediff_with_background
#                 - (albedo * concentration)[None, :, :] * target_spectrum[:, None, None]
#             )
#             covariance = np.tensordot(
#                 d_covariance, d_covariance, axes=((1, 2), (1, 2))
#             ) / (rows * cols)
#             covariance += np.eye(covariance.shape[0]) * 1e-6  # Regularization
#             covariance_inverse = np.linalg.inv(covariance)

#             # Step 7.3: Update common denominator and compute concentration
#             common_denominator = (
#                 target_spectrum.T @ covariance_inverse @ target_spectrum
#             )
#             numerator = (
#                 np.einsum(
#                     "ijk,i->jk",
#                     radiancediff_with_background,
#                     np.dot(covariance_inverse, target_spectrum),
#                 )
#                 - l1filter
#             )
#             concentration = np.maximum(numerator / (albedo * common_denominator), 0)

#     return concentration


def ml_matched_filter(
    data_cube: np.ndarray,
    initial_unit_absorption_spectrum: np.ndarray,
    uas_list: np.ndarray,
    transmittance_list: np.ndarray,
    iterate: bool = False,
    albedoadjust: bool = False,
    sparsity: bool = False,
    dynamic_adjust: bool = True,  # 新增动态调整标志
    threshold: float = 5000,  # 初始浓度增强阈值
    threshold_step: float = 5000,  # 阈值调整步长
    max_threshold: float = 16000,  # 最大浓度增强阈值
) -> np.ndarray:
    """Calculate the methane enhancement of the image data with iterative and dynamic adjustment.

    Args:
        data_cube (np.ndarray): 3D array representing the image data cube.
        unit_absorption_spectrum (np.ndarray): 1D array representing the unit absorption spectrum.
        iterate (bool): Flag indicating whether to perform iterative computation.
        albedoadjust (bool): Flag indicating whether to adjust for albedo.
        dynamic_adjust (bool): Flag indicating whether to perform dynamic adjustment for high concentrations.
        threshold (float): Initial threshold for dynamic adjustment.
        threshold_step (float): Step size for increasing the threshold.
        max_threshold (float): Maximum threshold for dynamic adjustment.

    Returns:
        np.ndarray: 2D array representing the concentration of methane.
    """
    # Ensure data_cube is a 3D array
    if data_cube.ndim == 2:
        data_cube = data_cube[np.newaxis, :, :]

    bands, rows, cols = data_cube.shape
    concentration = np.zeros((rows, cols))

    # Step 1: Calculate background spectrum and target spectrum
    background_spectrum = np.nanmean(
        data_cube, axis=(1, 2)
    )  # Mean across rows and cols
    target_spectrum = background_spectrum * initial_unit_absorption_spectrum

    # Step 2: Calculate radiance difference (handling NaNs)
    radiancediff_with_background = data_cube - background_spectrum[:, None, None]

    # Step 3: Compute covariance matrix (avoid NaNs)
    d_covariance = radiancediff_with_background
    covariance = np.tensordot(d_covariance, d_covariance, axes=((1, 2), (1, 2))) / (
        rows * cols
    )
    covariance += np.eye(covariance.shape[0]) * 1e-6  # Regularization
    covariance_inverse = np.linalg.inv(covariance)

    # Step 4: Adjust albedo if needed
    albedo = np.ones((rows, cols))
    if albedoadjust:
        albedo = np.einsum("ijk,i->jk", data_cube, background_spectrum) / np.dot(
            background_spectrum, background_spectrum
        )
        albedo = np.nan_to_num(albedo, nan=1.0)

    # Step 5: Precompute common denominator
    common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum

    # Step 6: Compute concentration (vectorized)
    numerator = np.einsum(
        "ijk,i->jk",
        radiancediff_with_background,
        np.dot(covariance_inverse, target_spectrum),
    )
    concentration = numerator / (albedo * common_denominator)

    # Step 7: Handle iteration for more accurate concentration calculation
    if iterate:
        l1filter = np.zeros((rows, cols))
        for _ in range(5):
            # Step 7.0: Handle sparsity
            if sparsity:
                l1filter = 1 / (concentration + 1e-6)

            # Step 7.1: Update residual (background and target spectra)
            residual = (
                data_cube
                - (albedo * concentration)[None, :, :] * target_spectrum[:, None, None]
            )

            background_spectrum = np.nanmean(residual, axis=(1, 2))

            target_spectrum = background_spectrum * initial_unit_absorption_spectrum

            # Step 7.2: Recompute radiance difference and covariance
            radiancediff_with_background = (
                data_cube - background_spectrum[:, None, None]
            )

            d_covariance = (
                radiancediff_with_background
                - (albedo * concentration)[None, :, :] * target_spectrum[:, None, None]
            )
            covariance = np.tensordot(
                d_covariance, d_covariance, axes=((1, 2), (1, 2))
            ) / (rows * cols)
            covariance += np.eye(covariance.shape[0]) * 1e-6  # Regularization
            covariance_inverse = np.linalg.inv(covariance)

            # Step 7.3: Update common denominator and compute concentration
            common_denominator = (
                target_spectrum.T @ covariance_inverse @ target_spectrum
            )
            numerator = (
                np.einsum(
                    "ijk,i->jk",
                    radiancediff_with_background,
                    np.dot(covariance_inverse, target_spectrum),
                )
                - l1filter
            )
            concentration = np.maximum(numerator / (albedo * common_denominator), 0)

    i = 0
    if dynamic_adjust:
        # Handle high-concentration pixels using multi-step adjustment
        high_conc_mask = concentration > threshold
        while np.any(high_conc_mask) and threshold < max_threshold:
            current_background_spectrum = background_spectrum * transmittance_list[i]
            current_target_spectrum = (
                current_background_spectrum * uas_list[i]
            )  # Example dynamic selection
            radiancediff_with_background = (
                data_cube - current_background_spectrum[:, None, None]
            )
            # Recompute concentration for high-concentration pixels
            numerator = np.einsum(
                "ijk,i->jk",
                radiancediff_with_background,
                np.dot(covariance_inverse, current_target_spectrum),
            )
            common_denominator = (
                current_target_spectrum.T @ covariance_inverse @ current_target_spectrum
            )
            new_concentration = numerator / (albedo * common_denominator) + threshold

            # Update concentration and threshold
            concentration[high_conc_mask] = new_concentration[high_conc_mask]
            threshold += threshold_step
            i += 1
            high_conc_mask = concentration > threshold

    return concentration


# def ml_matched_filter(
#     data_cube: np.ndarray,
#     unit_absorption_spectrum: np.ndarray,
#     satellitetype: str,
#     iterate: bool,
#     albedoadjust: bool,
#     sparsity: bool = False,
# ) -> np.ndarray:
#     # estimate the whole range of methane concentration

#     # Ensure data_cube is a 3D array
#     if data_cube.ndim == 2:
#         data_cube = data_cube[np.newaxis, :, :]

#     bands, rows, cols = data_cube.shape
#     concentration = np.zeros((rows, cols))

#     # Step 1: Calculate background spectrum and target spectrum
#     background_spectrum = np.nanmean(
#         data_cube, axis=(1, 2)
#     )  # Mean across rows and cols
#     target_spectrum = background_spectrum * unit_absorption_spectrum

#     # Step 2: Calculate radiance difference (handling NaNs)
#     radiancediff_with_background = data_cube - background_spectrum[:, None, None]

#     # Step 3: Compute covariance matrix (avoid NaNs)
#     d_covariance = radiancediff_with_background
#     covariance = np.tensordot(d_covariance, d_covariance, axes=((1, 2), (1, 2))) / (
#         rows * cols
#     )
#     covariance += np.eye(covariance.shape[0]) * 1e-6  # Regularization
#     covariance_inverse = np.linalg.inv(covariance)

#     # Step 4: Adjust albedo if needed
#     albedo = np.ones((rows, cols))
#     if albedoadjust:
#         albedo = np.einsum("ijk,i->jk", data_cube, background_spectrum) / np.dot(
#             background_spectrum, background_spectrum
#         )
#         albedo = np.nan_to_num(albedo, nan=1.0)

#     # Step 5: Precompute common denominator
#     common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum

#     # Step 6: Compute concentration (vectorized)
#     numerator = np.einsum(
#         "ijk,i->jk",
#         radiancediff_with_background,
#         np.dot(covariance_inverse, target_spectrum),
#     )
#     concentration = numerator / (albedo * common_denominator)

#     std = np.std(concentration)
#     mean = np.mean(concentration)
#     max = np.max(concentration)

#     adaptive_threshold = mean + 2 * std

#     low_concentration_mask = concentration < adaptive_threshold
#     _, bg_uas = glut.generate_satellite_uas_for_specific_range_from_lut(
#         satellitetype, 0, adaptive_threshold, 2150, 2500, 25, 0
#     )
#     # Step 7.1: Update residual (background and target spectra)
#     residual = (
#         data_cube
#         - (albedo * concentration)[None, :, :] * target_spectrum[:, None, None]
#     )

#     background_spectrum = np.nanmean(residual, axis=(1, 2))
#     target_spectrum = background_spectrum * bg_uas

#     # Step 7.2: Recompute radiance difference and covariance
#     radiancediff_with_background = data_cube - background_spectrum[:, None, None]

#     d_covariance = (
#         radiancediff_with_background
#         - (albedo * concentration)[None, :, :] * target_spectrum[:, None, None]
#     )
#     covariance = np.tensordot(d_covariance, d_covariance, axes=((1, 2), (1, 2))) / (
#         rows * cols
#     )
#     covariance += np.eye(covariance.shape[0]) * 1e-6  # Regularization
#     covariance_inverse = np.linalg.inv(covariance)

#     # Step 7.3: Update common denominator and compute concentration
#     common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum
#     numerator = np.einsum(
#         "ijk,i->jk",
#         radiancediff_with_background,
#         np.dot(covariance_inverse, target_spectrum),
#     )
#     concentration[low_concentration_mask] = np.maximum(
#         numerator / (albedo * common_denominator), 0
#     )[low_concentration_mask]

#     # if max > 40000:
#     #     threshold_interval = 10000
#     # elif max > 20000:
#     #     threshold_interval = 5000
#     # elif max > 10000:
#     #     threshold_interval = 2000
#     # elif max > 5000:
#     #     threshold_interval = 1000

#     return concentration


# def ml_matched_filter(
#     data_cube: np.ndarray,
#     unit_absorption_spectrum_array: np.ndarray,
#     albedoadjust: bool,
# ) -> np.ndarray:
#     """
#     Calculate the methane enhancement of the image data based on the modified matched filter
#     and the unit absorption spectrum.

#     :param data_cube: numpy array of the image data
#     :param unit_absorption_spectrum: list of the unit absorption spectrum numpy array from differenet range
#     :param albedoadjust: bool, whether to adjust the albedo

#     :return: numpy array of methane enhancement result
#     """
#     # 对背景进行固定并使用小范围UAS进行计算

#     concentration = mf(
#         data_cube, unit_absorption_spectrum_array[0], False, False, False
#     )

#     bg_threshold = np.percentile(concentration, 75)
#     if bg_threshold < 500:
#         bg_threshold = 500
#     _, bg_uas = glut.generate_satellite_uas_for_specific_range_from_lut(
#         "AHSI", 0, bg_threshold, 2150, 2500, 25, 0
#     )

#     def white_bg(
#         data_cube,
#         unit_absorption_spectrum,
#         iterate: bool,
#         albedoadjust: bool,
#         sparsity: bool = False,
#     ):
#         # Ensure data_cube is a 3D array
#         if data_cube.ndim == 2:
#             data_cube = data_cube[np.newaxis, :, :]

#         bands, rows, cols = data_cube.shape
#         concentration = np.zeros((rows, cols))

#         # Step 1: Calculate background spectrum and target spectrum
#         background_spectrum = np.nanmean(
#             data_cube, axis=(1, 2)
#         )  # Mean across rows and cols
#         target_spectrum = background_spectrum * unit_absorption_spectrum

#         # Step 2: Calculate radiance difference (handling NaNs)
#         radiancediff_with_background = data_cube - background_spectrum[:, None, None]

#         # Step 3: Compute covariance matrix (avoid NaNs)
#         d_covariance = radiancediff_with_background
#         covariance = np.tensordot(d_covariance, d_covariance, axes=((1, 2), (1, 2))) / (
#             rows * cols
#         )
#         covariance += np.eye(covariance.shape[0]) * 1e-6  # Regularization
#         covariance_inverse = np.linalg.inv(covariance)

#         # Step 4: Adjust albedo if needed
#         albedo = np.ones((rows, cols))
#         if albedoadjust:
#             albedo = np.einsum("ijk,i->jk", data_cube, background_spectrum) / np.dot(
#                 background_spectrum, background_spectrum
#             )
#             albedo = np.nan_to_num(albedo, nan=1.0)

#         # Step 5: Precompute common denominator
#         common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum

#         # Step 6: Compute concentration (vectorized)
#         numerator = np.einsum(
#             "ijk,i->jk",
#             radiancediff_with_background,
#             np.dot(covariance_inverse, target_spectrum),
#         )
#         concentration = numerator / (albedo * common_denominator)

#         # Step 7: Handle iteration for more accurate concentration calculation
#         if iterate:
#             l1filter = np.zeros((rows, cols))
#             for _ in range(5):
#                 # Step 7.0: Handle sparsity
#                 if sparsity:
#                     l1filter = 1 / (concentration + 1e-6)

#                 # Step 7.1: Update residual (background and target spectra)
#                 residual = (
#                     data_cube
#                     - (albedo * concentration)[None, :, :]
#                     * target_spectrum[:, None, None]
#                 )

#                 background_spectrum = np.nanmean(residual, axis=(1, 2))

#                 target_spectrum = background_spectrum * unit_absorption_spectrum

#                 # Step 7.2: Recompute radiance difference and covariance
#                 radiancediff_with_background = (
#                     data_cube - background_spectrum[:, None, None]
#                 )

#                 d_covariance = (
#                     radiancediff_with_background
#                     - (albedo * concentration)[None, :, :]
#                     * target_spectrum[:, None, None]
#                 )
#                 covariance = np.tensordot(
#                     d_covariance, d_covariance, axes=((1, 2), (1, 2))
#                 ) / (rows * cols)
#                 covariance += np.eye(covariance.shape[0]) * 1e-6  # Regularization
#                 covariance_inverse = np.linalg.inv(covariance)

#                 # Step 7.3: Update common denominator and compute concentration
#                 common_denominator = (
#                     target_spectrum.T @ covariance_inverse @ target_spectrum
#                 )
#                 numerator = (
#                     np.einsum(
#                         "ijk,i->jk",
#                         radiancediff_with_background,
#                         np.dot(covariance_inverse, target_spectrum),
#                     )
#                     - l1filter
#                 )
#                 concentration = np.maximum(numerator / (albedo * common_denominator), 0)

#         return concentration, background_spectrum

#     concentration, background_spectrum, target_spectrum = white_bg(
#         data_cube, bg_uas, True, False, False
#     )

#     # Ensure data_cube is a 3D array
#     if data_cube.ndim == 2:
#         data_cube = data_cube[np.newaxis, :, :]

#     # Step 1: Calculate background spectrum and target spectrum
#     background_spectrum = np.nanmean(
#         data_cube, axis=(1, 2)
#     )  # Mean across rows and cols
#     target_spectrum = background_spectrum * unit_absorption_spectrum_array[0]

#     background_spectrum = np.nanmean(
#         data_cube - concentration * target_spectrum, axis=(1, 2)
#     )

#     # Step 3: Compute covariance matrix (avoid NaNs)
#     d_covariance = radiancediff_with_background
#     covariance = np.tensordot(d_covariance, d_covariance, axes=((1, 2), (1, 2))) / (
#         rows * cols
#     )
#     covariance += np.eye(covariance.shape[0]) * 1e-6  # Regularization
#     covariance_inverse = np.linalg.inv(covariance)

#     # Step 4: Adjust albedo if needed
#     albedo = np.ones((rows, cols))
#     if albedoadjust:
#         albedo = np.einsum("ijk,i->jk", data_cube, background_spectrum) / np.dot(
#             background_spectrum, background_spectrum
#         )
#         albedo = np.nan_to_num(albedo, nan=1.0)

#     # Step 5: Precompute common denominator
#     common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum

#     # Step 6: Compute concentration (vectorized)
#     numerator = np.einsum(
#         "ijk,i->jk",
#         radiancediff_with_background,
#         np.dot(covariance_inverse, target_spectrum),
#     )
#     concentration = numerator / (albedo * common_denominator)

#     # 多层单位吸收光谱计算
#     levelon = True
#     adaptive_threshold = 6000
#     i = 1
#     high_concentration_mask = concentration > adaptive_threshold

#     while levelon:
#         if np.sum(high_concentration_mask) > 0 and adaptive_threshold < 32000:
#             background_spectrum = np.nanmean(
#                 data_cube - concentration * target_spectrum[:, None, None], axis=(1, 2)
#             )
#             target_spectrum = background_spectrum * unit_absorption_spectrum_array[0]

#             new_background_spectrum = background_spectrum
#             for n in range(i):
#                 new_background_spectrum += (
#                     6000 * new_background_spectrum * unit_absorption_spectrum_array[n]
#                 )
#             high_target_spectrum = (
#                 new_background_spectrum * unit_absorption_spectrum_array[i]
#             )

#             radiancediff_with_background[:, high_concentration_mask] = (
#                 data_cube[:, high_concentration_mask] - new_background_spectrum[:, None]
#             )
#             # radiancediff_with_background[:, low_concentration_mask] = (
#             #     data_cube[:, low_concentration_mask] - background_spectrum[:, None]
#             # )

#             # d_covariance[:,high_concentration_mask] = data_cube[:,high_concentration_mask] - (
#             #     (concentration[high_concentration_mask]-adaptive_threshold)*high_target_spectrum[:,None] + new_background_spectrum[:,None]
#             # )
#             # d_covariance[:,high_concentration_mask] = data_cube[:,high_concentration_mask] - (
#             #     background_spectrum[:,None] + concentration[high_concentration_mask]*target_spectrum[:,None]
#             # )

#             # d_covariance[:,low_concentration_mask] = data_cube[:,low_concentration_mask] - (
#             #     background_spectrum[:,None] + concentration[low_concentration_mask]*target_spectrum[:,None]
#             # )
#             # covariance = np.zeros((bands, bands))
#             # for row in range(rows):
#             #     for col in range(cols):
#             #         covariance += np.outer(d_covariance[:, row, col], d_covariance[:, row, col])
#             # covariance /= rows*cols
#             # covariance_inverse = np.linalg.pinv(covariance)

#             concentration[high_concentration_mask] = (
#                 (
#                     radiancediff_with_background[:, high_concentration_mask].T
#                     @ high_target_spectrum
#                 )
#                 / (high_target_spectrum.T @ high_target_spectrum)
#             ) + adaptive_threshold

#             adaptive_threshold += 6000
#             high_concentration_mask = concentration > adaptive_threshold * 0.99
#             i += 1

#         else:
#             levelon = False

#     return original_concentration, concentration


def ml_matched_filter_simulation_test():
    plume = np.load(
        r"/home/emeric/Documents/GitHub/MF/data/simulated_plumes/gaussianplume_1000_2_stability_D.npy"
    )
    simulated_radiance_cube = si.simulate_satellite_images_with_plume(
        "AHSI", plume, 25, 0, 2150, 2500, 0.01
    )

    uaslist = []
    transmittance_list = []
    # 设置初始化的单位吸收光谱范围
    _, initial_uas = glut.generate_satellite_uas_for_specific_range_from_lut(
        "AHSI", 0, 50000, 2150, 2500, 25, 0
    )
    for i in range(0, 50000, 5000):
        _, uas = glut.generate_satellite_uas_for_specific_range_from_lut(
            "AHSI", i, i + 5000, 2150, 2500, 25, 0
        )
        _, transmittance = glut.generate_transmittance_for_specific_range_from_lut(
            "AHSI", 0, i, 2150, 2500, 25, 0
        )
        uaslist.append(uas)
        transmittance_list.append(transmittance)

    startime = time.time()

    methane_concentration = ml_matched_filter(
        simulated_radiance_cube,
        initial_uas,
        uaslist,
        transmittance_list,
        True,
        False,
        False,
        True,
    )
    finish_time = time.time()
    print(f"running time: {finish_time - startime}")

    # 计算统计信息
    mean_concentration = np.mean(methane_concentration)
    std_concentration = np.std(methane_concentration)
    max_concentration = np.max(methane_concentration)
    min_concentration = np.min(methane_concentration)
    print(f"Mean: {mean_concentration:.2f} ppm")
    print(f"Std: {std_concentration:.2f} ppm")
    print(f"Max: {max_concentration:.2f} ppm")
    print(f"Min: {min_concentration:.2f} ppm")

    # 创建图形和子图
    fig, axes = plt.subplots(1, 3, figsize=(30, 12))
    # 子图1：甲烷浓度二维数组的可视化
    im = axes[0].imshow(methane_concentration, cmap="viridis", origin="lower")
    axes[0].set_title("Methane Concentration Enhancement (2D)")
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Y Coordinate")

    # 将 colorbar 移到下方
    cbar = fig.colorbar(
        im, ax=axes[0], orientation="horizontal", shrink=0.7, fraction=0.046, pad=0.04
    )
    cbar.set_label("Methane Concentration (ppm)")

    # 在第一个子图上添加统计信息
    stats_text = (
        f"Mean: {mean_concentration:.2f} ppm\n"
        f"Std: {std_concentration:.2f} ppm\n"
        f"Max: {max_concentration:.2f} ppm\n"
        f"Min: {min_concentration:.2f} ppm"
    )
    axes[0].text(
        1.05,
        0.5,
        stats_text,
        transform=axes[0].transAxes,
        fontsize=12,
        va="center",
        bbox=dict(facecolor="white", alpha=0.6),
    )

    # 子图2：甲烷浓度分布的直方图和 KDE 图

    plume_mask = plume < 100
    sns.histplot(
        methane_concentration[plume_mask].flatten() - plume[plume_mask].flatten(),
        bins=100,
        kde=True,
        ax=axes[1],
    )
    axes[1].set_title("Distribution of Methane Concentration")
    axes[1].set_xlabel("Methane Concentration (ppm)")
    axes[1].set_ylabel("Frequency")

    plume_mask = plume > 100
    axes[2].plot(
        plume[plume_mask].flatten(), methane_concentration[plume_mask].flatten(), "o"
    )
    axes[2].set_title("Scatter plot of Methane Concentration")
    axes[2].set_xlabel("Plume Concentration (ppm)")
    axes[2].set_ylabel("Methane Concentration (ppm)")
    # 调整布局
    fig.tight_layout()
    # 显示图表
    plt.show()

    return


def ml_matched_filter_real_image_test():
    filepath = "C:\\Users\\RS\\Desktop\\Lifei_essay_data\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222_SW.tif"
    outputfolder = "C:\\Users\\RS\\Desktop\\Lifei_essay_data\\Lifei_essay_result\\"
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return

    _, image_cube = sd.GF5B_data.get_calibrated_radiance(filepath, 2150, 2500)
    # 取整幅影像的 100*100 切片进行测试
    image_sample_cube = image_cube[:, 500:600, 700:800]
    _, unit_absorption_spectrum = (
        glut.generate_satellite_uas_for_specific_range_from_lut(
            "AHSI", 0, 50000, 2150, 2500, 25, 0
        )
    )
    startime = time.time()
    methane_concentration = ml_matched_filter(
        image_sample_cube, unit_absorption_spectrum, "AHSI", False, False, False
    )
    finish_time = time.time()
    print(f"running time: {finish_time - startime}")

    # sd.AHSI_data.export_ahsi_array_to_tiff(
    #     methane_concentration,
    #     filepath,
    #     outputfolder,
    # )

    # 计算统计信息
    mean_concentration = np.mean(methane_concentration)
    std_concentration = np.std(methane_concentration)
    max_concentration = np.max(methane_concentration)
    min_concentration = np.min(methane_concentration)
    print(f"Mean: {mean_concentration:.2f} ppm")
    print(f"Std: {std_concentration:.2f} ppm")
    print(f"Max: {max_concentration:.2f} ppm")
    print(f"Min: {min_concentration:.2f} ppm")

    # 创建图形和子图
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # 子图1：甲烷浓度二维数组的可视化
    im = axes[0].imshow(methane_concentration, cmap="viridis", origin="lower")
    axes[0].set_title("Methane Concentration Enhancement (2D)")
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Y Coordinate")

    # 将 colorbar 移到下方
    cbar = fig.colorbar(
        im, ax=axes[0], orientation="horizontal", shrink=0.7, fraction=0.046, pad=0.04
    )
    cbar.set_label("Methane Concentration (ppm)")

    # 在第一个子图上添加统计信息
    stats_text = (
        f"Mean: {mean_concentration:.2f} ppm\n"
        f"Std: {std_concentration:.2f} ppm\n"
        f"Max: {max_concentration:.2f} ppm\n"
        f"Min: {min_concentration:.2f} ppm"
    )
    axes[0].text(
        1.05,
        0.5,
        stats_text,
        transform=axes[0].transAxes,
        fontsize=12,
        va="center",
        bbox=dict(facecolor="white", alpha=0.6),
    )

    # 子图2：甲烷浓度分布的直方图和 KDE 图
    sns.histplot(methane_concentration.flatten(), bins=50, kde=True, ax=axes[1])
    axes[1].set_title("Distribution of Methane Concentration")
    axes[1].set_xlabel("Methane Concentration (ppm)")
    axes[1].set_ylabel("Frequency")

    # 调整布局
    fig.tight_layout()
    # 显示图表
    plt.show()

    return


if __name__ == "__main__":
    ml_matched_filter_simulation_test()
    # ml_matched_filter_real_image_test()
