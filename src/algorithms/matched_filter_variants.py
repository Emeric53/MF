import numpy as np

import os
import sys
import time

sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src")

from utils.satellites_data import general_functions as gf
from utils import satellites_data as sd
from utils import generate_radiance_lut_and_uas as glut


def compute_important_parameters(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    polluted: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the background spectrum and target spectrum and the radiance diff with background and the d_covariance
    """
    if polluted is not None:
        if data_cube.ndim == 3:
            background_spectrum = np.nanmean(data_cube - polluted, axis=(1, 2))
            target_spectrum = background_spectrum * unit_absorption_spectrum
            radiancediff_with_background = (
                data_cube - background_spectrum[:, None, None]
            )
        elif data_cube.ndim == 2:
            background_spectrum = np.nanmean(data_cube - polluted, axis=1)
            target_spectrum = background_spectrum * unit_absorption_spectrum
            radiancediff_with_background = data_cube - background_spectrum[:, None]
        else:
            raise ValueError("Data cube must be 2D or 3D")
    else:
        if data_cube.ndim == 3:
            background_spectrum = np.nanmean(data_cube, axis=(1, 2))
            target_spectrum = background_spectrum * unit_absorption_spectrum
            radiancediff_with_background = (
                data_cube - background_spectrum[:, None, None]
            )
        elif data_cube.ndim == 2:
            background_spectrum = np.nanmean(data_cube, axis=1)
            target_spectrum = background_spectrum * unit_absorption_spectrum
            radiancediff_with_background = data_cube - background_spectrum[:, None]
        else:
            raise ValueError("Data cube must be 2D or 3D")
    return (
        background_spectrum,
        target_spectrum,
        radiancediff_with_background,
    )


def compute_covariance_inverse_and_common_denominator(
    radiancediff_with_background: np.ndarray,
    target_spectrum: np.ndarray,
    counts: int,
):
    """
    Compute the covariance matrix and its inverse.
    """
    if radiancediff_with_background.ndim == 3:
        covariance = (
            np.tensordot(
                radiancediff_with_background,
                radiancediff_with_background,
                axes=((1, 2), (1, 2)),
            )
            / counts
        )
    elif radiancediff_with_background.ndim == 2:
        covariance = (
            np.dot(radiancediff_with_background, radiancediff_with_background.T)
            / counts
        )
    else:
        raise ValueError("Radiancediff_with_background must be 2D or 3D")
    covariance_inverse = np.linalg.pinv(covariance)
    common_denominator = target_spectrum @ covariance_inverse @ target_spectrum
    return covariance_inverse, common_denominator


def compute_albedo(
    data_cube: np.ndarray, background_spectrum: np.ndarray
) -> np.ndarray:
    """
    Compute the albedo correction factor for either a 3D (full image) or 2D (single column) data cube.

    Args:
        data_cube (np.ndarray): The image data cube, either 3D (bands, rows, cols) or 2D (bands, rows).
        background_spectrum (np.ndarray): The background spectrum array for the image data.

    Returns:
        np.ndarray: The albedo correction factor.
    """
    # Check if the data_cube is 3D (full image) or 2D (single column)
    if data_cube.ndim == 3:  # 3D case: (bands, rows, cols)
        albedo = np.einsum("ijk,i->jk", data_cube, background_spectrum) / np.dot(
            background_spectrum, background_spectrum
        )
    elif data_cube.ndim == 2:  # 2D case: (bands, rows)
        albedo = np.einsum("ij,i->j", data_cube, background_spectrum) / np.dot(
            background_spectrum, background_spectrum
        )
    else:
        raise ValueError("Input data_cube must be either 2D or 3D.")

    return albedo


def compute_concentration_pixel(
    radiancediff_with_background: np.ndarray,
    covariance_inverse: np.ndarray,
    target_spectrum: np.ndarray,
    albedo: float,
    common_denominator: float,
) -> np.ndarray:
    """
    Compute concentration for a single pixel.
    """
    numerator = radiancediff_with_background.T @ covariance_inverse @ target_spectrum
    concentration = numerator / (albedo * common_denominator)
    return concentration


def compute_concentration_column(
    column_radiancediff_with_bg: np.ndarray,
    covariance_inverse: np.ndarray,
    target_spectrum: np.ndarray,
    column_albedo: np.ndarray,
    common_denominator: float,
) -> np.ndarray:
    # Initial concentration calculation
    numerator = column_radiancediff_with_bg.T @ covariance_inverse @ target_spectrum
    denominator = column_albedo * common_denominator
    column_concentration = numerator / denominator
    return column_concentration


def matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    iterate: bool = False,
    albedoadjust: bool = False,
) -> np.ndarray:
    """
    Calculate methane enhancement based on the original matched filter method.
    """
    # initiate
    _, rows, cols = data_cube.shape
    concentration = np.zeros((rows, cols))
    albedo = np.ones((rows, cols))

    # Compute initial background and target spectrum, radiance difference with background, and covariance inverse
    (
        background_spectrum,
        target_spectrum,
        radiancediff_with_background,
    ) = compute_important_parameters(data_cube, unit_absorption_spectrum)

    d_covariance = radiancediff_with_background
    covariance_inverse, common_denominator = (
        compute_covariance_inverse_and_common_denominator(
            d_covariance, target_spectrum, rows * cols
        )
    )
    # Compute albedo adjustment if required
    if albedoadjust:
        albedo = compute_albedo(data_cube, background_spectrum)

    # Compute concentration for each pixel
    for row in range(rows):
        for col in range(cols):
            concentration[row, col] = compute_concentration_pixel(
                radiancediff_with_background[:, row, col],
                covariance_inverse,
                target_spectrum,
                albedo[row, col],
                common_denominator,
            )

    # Perform iterative updates if requested
    if iterate:
        for _ in range(5):
            # Update background, target spectrum, radiance difference with background, and covariance inverse
            (
                background_spectrum,
                target_spectrum,
                radiancediff_with_background,
            ) = compute_important_parameters(
                data_cube,
                unit_absorption_spectrum,
                albedo[None, :, :]
                * concentration[None, :, :]
                * target_spectrum[:, None, None],
            )
            d_covariance = (
                radiancediff_with_background
                - albedo[None, :, :]
                * concentration[None, :, :]
                * target_spectrum[:, None, None]
            )

            covariance_inverse, common_denominator = (
                compute_covariance_inverse_and_common_denominator(
                    d_covariance, target_spectrum, rows * cols
                )
            )

            for row in range(rows):
                for col in range(cols):
                    concentration[row, col] = np.maximum(
                        compute_concentration_pixel(
                            radiancediff_with_background[:, row, col],
                            covariance_inverse,
                            target_spectrum,
                            albedo[row, col],
                            common_denominator,
                        ),
                        0,
                    )

    return concentration


def columnwise_matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    iterate: bool = False,
    albedoadjust: bool = False,
) -> np.ndarray:
    """
    Perform matched filter calculation column by column using multithreading.
    """

    """
    Process a single column of the data cube.
    """
    _, rows, cols = data_cube.shape
    concentration = np.nan * np.zeros((rows, cols))
    albedo = np.ones((rows, cols))

    for col_index in range(cols):
        current_column = data_cube[:, :, col_index]

        # Compute background and target spectrum and radiance difference with background and covariance inverse
        background_spectrum, target_spectrum, radiancediff_with_bg = (
            compute_important_parameters(current_column, unit_absorption_spectrum)
        )

        if albedoadjust:
            albedo[:, col_index] = compute_albedo(current_column, background_spectrum)

        d_covariance = radiancediff_with_bg
        covariance_inverse, common_denominator = (
            compute_covariance_inverse_and_common_denominator(
                d_covariance, target_spectrum, rows
            )
        )

        concentration[:, col_index] = compute_concentration_column(
            radiancediff_with_bg,
            covariance_inverse,
            target_spectrum,
            albedo[:, col_index],
            common_denominator,
        )
        # Iterative updates
        if iterate:
            for _ in range(5):
                (
                    background_spectrum,
                    target_spectrum,
                    radiancediff_with_bg,
                ) = compute_important_parameters(
                    current_column,
                    unit_absorption_spectrum,
                    (
                        albedo[:, col_index]
                        * concentration[:, col_index]
                        * target_spectrum[:, None]
                    ),
                )

                d_covariance = radiancediff_with_bg - (
                    albedo[:, col_index]
                    * concentration[:, col_index]
                    * target_spectrum[:, None]
                )
                covariance_inverse, common_denominator = (
                    compute_covariance_inverse_and_common_denominator(
                        d_covariance, target_spectrum, rows
                    )
                )

                concentration[:, col_index] = np.maximum(
                    compute_concentration_column(
                        radiancediff_with_bg,
                        covariance_inverse,
                        target_spectrum,
                        albedo[:, col_index],
                        common_denominator,
                    ),
                    0,
                )

    return concentration


# 多层计算匹配滤波算法
def ml_matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum_array: np.ndarray,
    albedoadjust: bool,
) -> np.ndarray:
    """
    Calculate the methane enhancement of the image data based on the modified matched filter
    and the unit absorption spectrum.

    :param data_cube: numpy array of the image data
    :param unit_absorption_spectrum: list of the unit absorption spectrum numpy array from differenet range
    :param albedoadjust: bool, whether to adjust the albedo

    :return: numpy array of methane enhancement result
    """
    # 获取波段 行数 列数 初始化 concentration 数组，大小与卫星数据尺寸一致
    _, rows, cols = data_cube.shape
    concentration = np.zeros((rows, cols))
    albedo = np.ones((rows, cols))

    # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
    (
        background_spectrum,
        target_spectrum,
        radiance_diff_with_background,
    ) = compute_important_parameters(data_cube, unit_absorption_spectrum_array[0])
    d_covariance = radiance_diff_with_background

    covariance_inverse, common_denominator = (
        compute_covariance_inverse_and_common_denominator(
            d_covariance, target_spectrum, rows * cols
        )
    )

    # 判断是否进行反照率校正，若是，则通过背景光谱和实际光谱计算反照率校正因子
    if albedoadjust:
        compute_albedo(data_cube, background_spectrum)

    for row in range(rows):
        for col in range(cols):
            concentration[row, col] = compute_concentration_pixel(
                radiance_diff_with_background[:, row, col],
                covariance_inverse,
                target_spectrum,
                albedo[row, col],
                common_denominator,
            )

    # 备份原始浓度值
    original_concentration = concentration.copy()

    # 多层单位吸收光谱计算
    levelon = True
    adaptive_threshold = 6000
    i = 1
    high_concentration_mask = original_concentration > adaptive_threshold * (0.99**i)
    low_concentration_mask = original_concentration <= adaptive_threshold * (0.99**i)

    while levelon:
        if np.sum(high_concentration_mask) > 0 and adaptive_threshold < 32000:
            (
                background_spectrum,
                target_spectrum,
                radiancediff_with_background,
            ) = compute_important_parameters(
                data_cube,
                unit_absorption_spectrum_array[0],
                albedo[None, :, :]
                * concentration[None, :, :]
                * target_spectrum[:, None, None],
            )
            d_covariance = radiancediff_with_background - (
                albedo[None, :, :]
                * concentration[None, :, :]
                * target_spectrum[:, None, None]
            )
            covariance_inverse, common_denominator = (
                compute_covariance_inverse_and_common_denominator(
                    d_covariance, target_spectrum, rows * cols
                )
            )

            new_background_spectrum = background_spectrum
            for n in range(i):
                new_background_spectrum += (
                    6000 * new_background_spectrum * unit_absorption_spectrum_array[n]
                )

            high_target_spectrum = (
                new_background_spectrum * unit_absorption_spectrum_array[i]
            )

            radiancediff_with_background[:, high_concentration_mask] = (
                data_cube[:, high_concentration_mask] - new_background_spectrum[:, None]
            )
            radiancediff_with_background[:, low_concentration_mask] = (
                data_cube[:, low_concentration_mask] - background_spectrum[:, None]
            )

            concentration[high_concentration_mask] = (
                compute_concentration_pixel(
                    radiancediff_with_background[:, high_concentration_mask],
                    covariance_inverse,
                    high_target_spectrum,
                    albedo[high_concentration_mask],
                    common_denominator,
                )
                + adaptive_threshold
            )

            high_concentration_mask = original_concentration > adaptive_threshold * 0.99
            low_concentration_mask = original_concentration <= adaptive_threshold * 0.99

            adaptive_threshold += 6000
            i += 1

        else:
            levelon = False

    return concentration


def ml_matched_filter_new(
    satellite_name,
    data_cube: np.ndarray,
    unit_absorption_spectrum_array: np.ndarray,
    albedoadjust: bool,
) -> np.ndarray:
    """
    Calculate the methane enhancement of the image data based on the modified matched filter
    and the unit absorption spectrum.

    :param data_cube: numpy array of the image data
    :param unit_absorption_spectrum: list of the unit absorption spectrum numpy array from differenet range
    :param albedoadjust: bool, whether to adjust the albedo

    :return: numpy array of methane enhancement result
    """
    # 获取波段 行数 列数 初始化 concentration 数组，大小与卫星数据尺寸一致
    _, rows, cols = data_cube.shape
    concentration = np.zeros((rows, cols))
    albedo = np.ones((rows, cols))

    # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
    (
        background_spectrum,
        target_spectrum,
        radiance_diff_with_background,
    ) = compute_important_parameters(data_cube, unit_absorption_spectrum_array[0])
    d_covariance = radiance_diff_with_background

    covariance_inverse, common_denominator = (
        compute_covariance_inverse_and_common_denominator(
            d_covariance, target_spectrum, rows * cols
        )
    )

    # 判断是否进行反照率校正，若是，则通过背景光谱和实际光谱计算反照率校正因子
    if albedoadjust:
        compute_albedo(data_cube, background_spectrum)

    for row in range(rows):
        for col in range(cols):
            concentration[row, col] = compute_concentration_pixel(
                radiance_diff_with_background[:, row, col],
                covariance_inverse,
                target_spectrum,
                albedo[row, col],
                common_denominator,
            )
    # 到这里为止 和 默认算法一致

    # 备份原始浓度值
    original_concentration = concentration.copy()

    std = np.std(concentration)
    mean = np.mean(concentration)
    std_mean = std + mean
    max = np.maximum(np.max(concentration) + std_mean, 50000)

    # 多层单位吸收光谱计算
    go_on = True
    adaptive_threshold = std_mean
    high_concentration_mask = original_concentration > adaptive_threshold
    low_concentration_mask = original_concentration <= adaptive_threshold
    low_concentration = np.zeros_like(concentration)
    high_concentration = np.zeros_like(concentration)
    while go_on:
        print(f"current threshold is {adaptive_threshold}")
        _, low_uas = glut.generate_satellite_uas_for_specific_range_from_lut(
            "AHSI", 0, adaptive_threshold, 2150, 2500, 25, 0
        )
        _, high_uas = glut.generate_satellite_uas_for_specific_range_from_lut(
            "AHSI", adaptive_threshold, max, 2150, 2500, 25, 0
        )

        (
            background_spectrum,
            target_spectrum,
            radiancediff_with_background,
        ) = compute_important_parameters(
            data_cube,
            low_uas,
            albedo[None, :, :]
            * concentration[None, :, :]
            * target_spectrum[:, None, None],
        )

        d_covariance = radiancediff_with_background - (
            albedo[None, :, :]
            * concentration[None, :, :]
            * target_spectrum[:, None, None]
        )

        covariance_inverse, common_denominator = (
            compute_covariance_inverse_and_common_denominator(
                d_covariance, target_spectrum, rows * cols
            )
        )

        for row in range(rows):
            for col in range(cols):
                low_concentration[row, col] = compute_concentration_pixel(
                    radiancediff_with_background[:, row, col],
                    covariance_inverse,
                    target_spectrum,
                    albedo[row, col],
                    common_denominator,
                )

        (
            background_spectrum,
            target_spectrum,
            radiancediff_with_background,
        ) = compute_important_parameters(
            data_cube,
            high_uas,
            albedo[None, :, :]
            * (low_concentration[None, :, :] + adaptive_threshold)
            * target_spectrum[:, None, None],
        )
        d_covariance = radiancediff_with_background - (
            albedo[None, :, :]
            * concentration[None, :, :]
            * target_spectrum[:, None, None]
        )

        covariance_inverse, common_denominator = (
            compute_covariance_inverse_and_common_denominator(
                d_covariance, target_spectrum, rows * cols
            )
        )

        for row in range(rows):
            for col in range(cols):
                high_concentration[row, col] = (
                    compute_concentration_pixel(
                        radiancediff_with_background[:, row, col],
                        covariance_inverse,
                        target_spectrum,
                        albedo[row, col],
                        common_denominator,
                    )
                    + adaptive_threshold
                )
        # concentration[high_concentration_mask] = high_concentration[
        #     high_concentration_mask
        # ]
        # concentration[low_concentration_mask] = low_concentration[
        #     low_concentration_mask
        # ]
        concentration = (high_concentration + low_concentration) / 2

        # high_concentration_mask = concentration > adaptive_threshold
        # low_concentration_mask = concentration <= adaptive_threshold
        std = np.std(concentration)
        mean = np.mean(concentration)
        std_mean = std + mean
        max = np.max(concentration)

        if np.abs((std_mean - adaptive_threshold) / adaptive_threshold) < 0.1:
            go_on = False
        else:
            adaptive_threshold = std_mean

    return concentration


# convert the radiance into log space 整幅图像进行计算
def lognormal_matched_filter(
    data_cube: np.ndarray, unit_absorption_spectrum: np.ndarray
):
    # 获取 以 波段 行数 列数 为顺序的数据
    bands, rows, cols = data_cube.shape
    # 初始化 concentration 数组，大小与卫星数据尺寸一致
    concentration = np.zeros((rows, cols))
    # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
    log_background_spectrum = np.nanmean(np.log(data_cube), axis=(1, 2))
    background_spectrum = np.exp(log_background_spectrum)

    # 对当前目标光谱的每一行进行去均值操作，得到调整后的光谱，以此为基础计算协方差矩阵，并获得其逆矩阵
    radiancediff_with_bg = np.log(data_cube) - log_background_spectrum[:, None, None]
    d_covariance = radiancediff_with_bg
    covariance_inverse, common_denominator = (
        compute_covariance_inverse_and_common_denominator(
            d_covariance, unit_absorption_spectrum, rows * cols
        )
    )

    for row in range(rows):
        for col in range(cols):
            # 基于最优化公式计算每个像素的甲烷浓度增强值
            numerator = (
                radiancediff_with_bg[:, row, col].T
                @ covariance_inverse
                @ unit_absorption_spectrum
            )
            concentration[row, col] = numerator / common_denominator
    return concentration


def matched_filter_test(*args):
    filepath = "C:\\Users\\RS\\Desktop\\GF5-02_李飞论文所用数据\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222_SW.tif"
    outputfolder = "C:\\Users\\RS\\Desktop\\GF5-02_李飞论文所用数据\\mf_result\\"
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return

    _, image_cube = sd.AHSI_data.get_calibrated_radiance(filepath, 2150, 2500)
    image_sample_cube = image_cube[:, 500:600, 700:800]
    unit_absoprtion_spectrum_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\uas_files\\AHSI_unit_absorption_spectrum.txt"

    _, unit_absoprtion_spectrum = gf.open_unit_absorption_spectrum(
        unit_absoprtion_spectrum_path, 2150, 2500
    )
    startime = time.time()

    methane_concentration = matched_filter(
        data_cube=image_sample_cube,
        unit_absorption_spectrum=unit_absoprtion_spectrum,
        iterate=args[0],
        albedoadjust=args[1],
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

    # # 创建图形和子图
    # fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # # 子图1：甲烷浓度二维数组的可视化
    # im = axes[0].imshow(methane_concentration, cmap='viridis', origin='lower')
    # axes[0].set_title("Methane Concentration Enhancement (2D)")
    # axes[0].set_xlabel("X Coordinate")
    # axes[0].set_ylabel("Y Coordinate")

    # # 将 colorbar 移到下方
    # cbar = fig.colorbar(im, ax=axes[0], orientation='horizontal', shrink=0.7, fraction=0.046, pad=0.04)
    # cbar.set_label("Methane Concentration (ppm)")

    # # 在第一个子图上添加统计信息
    # stats_text = (f"Mean: {mean_concentration:.2f} ppm\n"
    #             f"Std: {std_concentration:.2f} ppm\n"
    #             f"Max: {max_concentration:.2f} ppm\n"
    #             f"Min: {min_concentration:.2f} ppm")
    # axes[0].text(1.05, 0.5, stats_text, transform=axes[0].transAxes,
    #             fontsize=12, va='center', bbox=dict(facecolor='white', alpha=0.6))

    # # 子图2：甲烷浓度分布的直方图和 KDE 图
    # sns.histplot(methane_concentration.flatten(), bins=50, kde=True, ax=axes[1])
    # axes[1].set_title("Distribution of Methane Concentration")
    # axes[1].set_xlabel("Methane Concentration (ppm)")
    # axes[1].set_ylabel("Frequency")

    # # 调整布局
    # fig.tight_layout()

    # # 显示图表
    # plt.show()

    return


def columnwise_matched_filter_test(*args):
    filepath = "C:\\Users\\RS\\Desktop\\GF5-02_李飞论文所用数据\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222_SW.tif"
    outputfolder = "C:\\Users\\RS\\Desktop\\GF5-02_李飞论文所用数据\\mf_result\\"
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return

    _, image_cube = sd.AHSI_data.get_calibrated_radiance(filepath, 2150, 2500)
    image_sample_cube = image_cube[:, 500:600, 700:800]
    unit_absoprtion_spectrum_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\uas_files\\AHSI_unit_absorption_spectrum.txt"

    _, unit_absoprtion_spectrum = gf.open_unit_absorption_spectrum(
        unit_absoprtion_spectrum_path, 2150, 2500
    )
    startime = time.time()
    methane_concentration = columnwise_matched_filter(
        data_cube=image_sample_cube,
        unit_absorption_spectrum=unit_absoprtion_spectrum,
        iterate=args[0],
        albedoadjust=args[1],
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

    # # 创建图形和子图
    # fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # # 子图1：甲烷浓度二维数组的可视化
    # im = axes[0].imshow(methane_concentration, cmap='viridis', origin='lower')
    # axes[0].set_title("Methane Concentration Enhancement (2D)")
    # axes[0].set_xlabel("X Coordinate")
    # axes[0].set_ylabel("Y Coordinate")

    # # 将 colorbar 移到下方
    # cbar = fig.colorbar(im, ax=axes[0], orientation='horizontal', shrink=0.7, fraction=0.046, pad=0.04)
    # cbar.set_label("Methane Concentration (ppm)")

    # # 在第一个子图上添加统计信息
    # stats_text = (f"Mean: {mean_concentration:.2f} ppm\n"
    #             f"Std: {std_concentration:.2f} ppm\n"
    #             f"Max: {max_concentration:.2f} ppm\n"
    #             f"Min: {min_concentration:.2f} ppm")
    # axes[0].text(1.05, 0.5, stats_text, transform=axes[0].transAxes,
    #             fontsize=12, va='center', bbox=dict(facecolor='white', alpha=0.6))

    # # 子图2：甲烷浓度分布的直方图和 KDE 图
    # sns.histplot(methane_concentration.flatten(), bins=50, kde=True, ax=axes[1])
    # axes[1].set_title("Distribution of Methane Concentration")
    # axes[1].set_xlabel("Methane Concentration (ppm)")
    # axes[1].set_ylabel("Frequency")

    # # 调整布局
    # fig.tight_layout()

    # # 显示图表
    # plt.show()

    return


def ml_matched_filter_test(*args):
    filepath = "C:\\Users\\RS\\Desktop\\GF5-02_李飞论文所用数据\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222_SW.tif"
    outputfolder = "C:\\Users\\RS\\Desktop\\GF5-02_李飞论文所用数据\\mf_result\\"
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return

    _, image_cube = sd.AHSI_data.get_calibrated_radiance(filepath, 2150, 2500)
    image_sample_cube = image_cube[:, 500:600, 700:800]
    unit_absoprtion_spectrum_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\uas_files\\AHSI_unit_absorption_spectrum.txt"

    _, unit_absoprtion_spectrum = gf.open_unit_absorption_spectrum(
        unit_absoprtion_spectrum_path, 2150, 2500
    )
    startime = time.time()
    print(image_sample_cube.shape)
    print(unit_absoprtion_spectrum.shape)
    methane_concentration = ml_matched_filter(
        data_cube=image_sample_cube,
        unit_absorption_spectrum_array=unit_absoprtion_spectrum,
        albedoadjust=args[0],
    )
    print(methane_concentration.shape)
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

    # # 创建图形和子图
    # fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # # 子图1：甲烷浓度二维数组的可视化
    # im = axes[0].imshow(methane_concentration, cmap='viridis', origin='lower')
    # axes[0].set_title("Methane Concentration Enhancement (2D)")
    # axes[0].set_xlabel("X Coordinate")
    # axes[0].set_ylabel("Y Coordinate")

    # # 将 colorbar 移到下方
    # cbar = fig.colorbar(im, ax=axes[0], orientation='horizontal', shrink=0.7, fraction=0.046, pad=0.04)
    # cbar.set_label("Methane Concentration (ppm)")

    # # 在第一个子图上添加统计信息
    # stats_text = (f"Mean: {mean_concentration:.2f} ppm\n"
    #             f"Std: {std_concentration:.2f} ppm\n"
    #             f"Max: {max_concentration:.2f} ppm\n"
    #             f"Min: {min_concentration:.2f} ppm")
    # axes[0].text(1.05, 0.5, stats_text, transform=axes[0].transAxes,
    #             fontsize=12, va='center', bbox=dict(facecolor='white', alpha=0.6))

    # # 子图2：甲烷浓度分布的直方图和 KDE 图
    # sns.histplot(methane_concentration.flatten(), bins=50, kde=True, ax=axes[1])
    # axes[1].set_title("Distribution of Methane Concentration")
    # axes[1].set_xlabel("Methane Concentration (ppm)")
    # axes[1].set_ylabel("Frequency")

    # # 调整布局
    # fig.tight_layout()

    # # 显示图表
    # plt.show()

    return


def lognormal_matched_filter_test():
    filepath = "C:\\Users\\RS\\Desktop\\GF5-02_李飞论文所用数据\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222_SW.tif"
    outputfolder = "C:\\Users\\RS\\Desktop\\GF5-02_李飞论文所用数据\\mf_result\\"
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return

    _, image_cube = sd.AHSI_data.get_calibrated_radiance(filepath, 2150, 2500)
    image_sample_cube = image_cube[:, 500:600, 700:800]
    unit_absoprtion_spectrum_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\uas_files\\AHSI_unit_absorption_spectrum.txt"

    _, unit_absoprtion_spectrum = gf.open_unit_absorption_spectrum(
        unit_absoprtion_spectrum_path, 2150, 2500
    )
    startime = time.time()
    print(image_sample_cube.shape)
    print(unit_absoprtion_spectrum.shape)
    methane_concentration = lognormal_matched_filter(
        data_cube=image_sample_cube,
        unit_absorption_spectrum=unit_absoprtion_spectrum,
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

    # # 创建图形和子图
    # fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # # 子图1：甲烷浓度二维数组的可视化
    # im = axes[0].imshow(methane_concentration, cmap='viridis', origin='lower')
    # axes[0].set_title("Methane Concentration Enhancement (2D)")
    # axes[0].set_xlabel("X Coordinate")
    # axes[0].set_ylabel("Y Coordinate")

    # # 将 colorbar 移到下方
    # cbar = fig.colorbar(im, ax=axes[0], orientation='horizontal', shrink=0.7, fraction=0.046, pad=0.04)
    # cbar.set_label("Methane Concentration (ppm)")

    # # 在第一个子图上添加统计信息
    # stats_text = (f"Mean: {mean_concentration:.2f} ppm\n"
    #             f"Std: {std_concentration:.2f} ppm\n"
    #             f"Max: {max_concentration:.2f} ppm\n"
    #             f"Min: {min_concentration:.2f} ppm")
    # axes[0].text(1.05, 0.5, stats_text, transform=axes[0].transAxes,
    #             fontsize=12, va='center', bbox=dict(facecolor='white', alpha=0.6))

    # # 子图2：甲烷浓度分布的直方图和 KDE 图
    # sns.histplot(methane_concentration.flatten(), bins=50, kde=True, ax=axes[1])
    # axes[1].set_title("Distribution of Methane Concentration")
    # axes[1].set_xlabel("Methane Concentration (ppm)")
    # axes[1].set_ylabel("Frequency")

    # # 调整布局
    # fig.tight_layout()

    # # 显示图表
    # plt.show()

    return


if __name__ == "__main__":
    # matched_filter_test(False, False)
    # columnwise_matched_filter_test(False, False)
    ml_matched_filter_test(False)
    # lognormal_matched_filter_test()
