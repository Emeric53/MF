import numpy as np

from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor


# matched filter algorithm 整幅影像进行计算
def matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    iterate: bool,
    albedoadjust: bool,
    sparsity: bool,
) -> np.ndarray:
    """Calculate the methane enhancement of the image data based on the original matched filter method.

    Args:
        data_cube (np.ndarray): 3D array representing the image data cube.
        unit_absorption_spectrum (np.ndarray): 1D array representing the unit absorption spectrum.
        iterate (bool): Flag indicating whether to perform iterative computation.
        albedoadjust (bool): Flag indicating whether to adjust for albedo.
        sparsity (bool): Flag indicating whether to consider sparsity.

    Returns:
        np.ndarray: 2D array representing the concentration of methane.

    """
    # 获取波段 行数 列数，初始化 concentration 数组，大小与卫星数据尺寸一致
    _, rows, cols = data_cube.shape
    concentration = np.zeros((rows, cols))

    # 计算背景光谱和目标光谱
    background_spectrum = np.nanmean(data_cube, axis=(1, 2))
    target_spectrum = background_spectrum * unit_absorption_spectrum
    radiancediff_with_background = data_cube - background_spectrum[:, None, None]

    # 使用矩阵计算协方差矩阵
    covariance = np.tensordot(
        radiancediff_with_background,
        radiancediff_with_background,
        axes=((1, 2), (1, 2)),
    ) / (rows * cols)
    covariance_inverse = np.linalg.pinv(covariance)

    # 计算反照率校正项
    albedo = np.ones((rows, cols))
    if albedoadjust:
        albedo = np.einsum("ijk,i->jk", data_cube, background_spectrum) / np.dot(
            background_spectrum, background_spectrum
        )

    # 计算目标谱的通用分母项
    common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum

    # 使用并行计算浓度
    def compute_concentration(row, col):
        numerator = (
            radiancediff_with_background[:, row, col].T
            @ covariance_inverse
            @ target_spectrum
        )
        denominator = albedo[row, col] * common_denominator
        return numerator / denominator

    for row in range(rows):
        for col in range(cols):
            concentration[row, col] = compute_concentration(row, col)

    # 如果需要迭代
    if iterate:
        # tol = 1e-6
        # prev_concentration = np.copy(concentration)
        # 初始化 l1filter
        l1filter = np.zeros((rows, cols))
        for _ in range(5):
            # 计算稀疏性矫正项
            if sparsity:
                l1filter = 1 / (concentration + np.finfo(np.float64).tiny)

            # 更新背景光谱和目标光谱
            background_spectrum = np.mean(
                data_cube
                - (albedo * concentration)[None, :, :] * target_spectrum[:, None, None],
                axis=(1, 2),
            )
            target_spectrum = background_spectrum * unit_absorption_spectrum
            radiancediff_with_background = (
                data_cube - background_spectrum[:, None, None]
            )

            # 重新计算协方差矩阵
            covariance = np.tensordot(
                radiancediff_with_background,
                radiancediff_with_background,
                axes=((1, 2), (1, 2)),
            ) / (rows * cols)
            covariance_inverse = np.linalg.pinv(covariance)

            # 更新 common_denominator
            common_denominator = (
                target_spectrum.T @ covariance_inverse @ target_spectrum
            )

            for row in range(rows):
                for col in range(cols):
                    concentration[row, col] = np.max(
                        compute_concentration(row, col)
                        - l1filter[row, col] / common_denominator,
                        0,
                    )

            # # 检查迭代是否可以提前终止
            # diff = np.abs(concentration - prev_concentration).max()
            # if diff < tol:
            #     break
            # prev_concentration = np.copy(concentration)

    return concentration


# columnwise matched filter algorithm 逐列计算
def columnwise_matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    iterate: bool = False,
    albedoadjust: bool = False,
    sparsity: bool = False,
) -> np.ndarray:
    _, rows, cols = data_cube.shape
    concentration = np.zeros((rows, cols))

    # 使用多线程并行化处理列
    def process_column(col_index):
        current_column = data_cube[:, :, col_index]
        valid_rows = ~np.isnan(current_column[0, :])
        count_not_nan = np.count_nonzero(valid_rows)

        if count_not_nan == 0:
            return np.nan * np.zeros(rows)

        # 计算背景光谱和目标光谱 以及 与背景光谱的差值
        background_spectrum = np.nanmean(current_column, axis=1)
        target_spectrum = background_spectrum * unit_absorption_spectrum
        radiancediff_with_bg = (
            current_column[:, valid_rows] - background_spectrum[:, None]
        )

        # 计算协方差矩阵
        d_covariance = radiancediff_with_bg
        covariance = np.einsum("ij,kj->ik", d_covariance, d_covariance) / count_not_nan
        covariance_inverse = np.linalg.pinv(covariance)

        albedo = np.ones(rows)
        if albedoadjust:
            albedo[valid_rows] = (
                current_column[:, valid_rows].T @ background_spectrum
            ) / (background_spectrum.T @ background_spectrum)

        # 初始浓度计算
        numerator = radiancediff_with_bg.T @ covariance_inverse @ target_spectrum
        denominator = albedo[valid_rows] * (
            target_spectrum.T @ covariance_inverse @ target_spectrum
        )
        conc = np.zeros(rows)
        conc[valid_rows] = numerator / denominator

        # 迭代更新
        if iterate:
            l1filter = np.zeros(rows)
            epsilon = np.finfo(np.float64).tiny
            for _ in range(5):
                if sparsity:
                    l1filter[valid_rows] = 1 / (conc[valid_rows] + epsilon)

                background_spectrum = np.nanmean(
                    current_column[:, valid_rows]
                    - albedo[valid_rows] * conc[valid_rows] * target_spectrum[:, None],
                    axis=1,
                )
                target_spectrum = background_spectrum * unit_absorption_spectrum
                radiancediff_with_bg = (
                    current_column[:, valid_rows] - background_spectrum[:, None]
                )

                d_covariance = current_column[:, valid_rows] - (
                    albedo[valid_rows] * conc[valid_rows] * target_spectrum[:, None]
                    + background_spectrum[:, None]
                )
                covariance = (
                    np.einsum("ij,ik->jk", d_covariance, d_covariance) / count_not_nan
                )
                covariance_inverse = np.linalg.pinv(covariance)

                numerator = (
                    radiancediff_with_bg.T @ covariance_inverse @ target_spectrum
                ) - l1filter[valid_rows]
                denominator = albedo[valid_rows] * (
                    target_spectrum.T @ covariance_inverse @ target_spectrum
                )
                conc[valid_rows] = np.maximum(numerator / denominator, 0.0)

        return conc

    # 并行执行列处理
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_column, range(cols)))

    concentration = np.stack(results, axis=1)
    return concentration


# multi-layer matched filter algorithm 整幅影像进行计算
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
    bands, rows, cols = data_cube.shape
    concentration = np.zeros((rows, cols))

    # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
    background_spectrum = np.nanmean(data_cube, axis=(1, 2))
    target_spectrum = background_spectrum * unit_absorption_spectrum_array[0]
    radiancediff_with_background = data_cube - background_spectrum[:, None, None]

    # 计算协方差矩阵，并获得其逆矩阵
    d_covariance = radiancediff_with_background

    # 使用矩阵计算协方差矩阵
    covariance = np.tensordot(
        d_covariance,
        d_covariance,
        axes=((1, 2), (1, 2)),
    ) / (rows * cols)
    covariance_inverse = np.linalg.pinv(covariance)

    # 判断是否进行反照率校正，若是，则通过背景光谱和实际光谱计算反照率校正因子
    albedo = np.ones((rows, cols))
    if albedoadjust:
        for row in range(rows):
            for col in range(cols):
                albedo[row, col] = (data_cube[:, row, col].T @ background_spectrum) / (
                    background_spectrum.T @ background_spectrum
                )

    # 基于最优化公式计算每个像素的甲烷浓度增强值
    common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum
    for row in range(rows):
        for col in range(cols):
            numerator = (
                radiancediff_with_background[:, row, col].T
                @ covariance_inverse
                @ target_spectrum
            )
            denominator = albedo[row, col] * common_denominator
            concentration[row, col] = numerator / denominator

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
            background_spectrum = np.nanmean(
                data_cube - concentration * target_spectrum[:, None, None], axis=(1, 2)
            )
            target_spectrum = background_spectrum * unit_absorption_spectrum_array[0]

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

            # d_covariance[:,high_concentration_mask] = data_cube[:,high_concentration_mask] - (
            #     (concentration[high_concentration_mask]-adaptive_threshold)*high_target_spectrum[:,None] + new_background_spectrum[:,None]
            # )
            # d_covariance[:,high_concentration_mask] = data_cube[:,high_concentration_mask] - (
            #     background_spectrum[:,None] + concentration[high_concentration_mask]*target_spectrum[:,None]
            # )

            # d_covariance[:,low_concentration_mask] = data_cube[:,low_concentration_mask] - (
            #     background_spectrum[:,None] + concentration[low_concentration_mask]*target_spectrum[:,None]
            # )
            # covariance = np.zeros((bands, bands))
            # for row in range(rows):
            #     for col in range(cols):
            #         covariance += np.outer(d_covariance[:, row, col], d_covariance[:, row, col])
            # covariance /= rows*cols
            # covariance_inverse = np.linalg.pinv(covariance)

            # concentration[high_concentration_mask] = (
            #     (radiancediff_with_bg[:, high_concentration_mask].T @ covariance_inverse @high_target_spectrum) /
            #     (high_target_spectrum.T @ covariance_inverse @ high_target_spectrum)
            # ) + adaptive_threshold

            concentration[high_concentration_mask] = (
                (
                    radiancediff_with_background[:, high_concentration_mask].T
                    @ high_target_spectrum
                )
                / (high_target_spectrum.T @ high_target_spectrum)
            ) + adaptive_threshold

            high_concentration_mask = original_concentration > adaptive_threshold * 0.99
            low_concentration_mask = original_concentration <= adaptive_threshold * 0.99

            adaptive_threshold += 6000
            i += 1

        else:
            levelon = False

    return original_concentration, concentration


# multi-layer matched filter algorithm 整幅影像进行计算
def ml_matched_filter2(
    data_cube: np.ndarray, unit_absorption_spectrum_list: list, albedoadjust: bool
) -> np.ndarray:
    """
    Calculate the methane enhancement of the image data based on the modified multi-layer matched filter
    and the unit absorption spectra.

    :param data_cube: numpy array of the image data
    :param unit_absorption_spectra: list of unit absorption spectrum numpy arrays from different ranges
    :param albedoadjust: bool, whether to adjust the albedo

    :return: numpy array of methane enhancement result (original and final concentration)
    """
    # Get bands, rows, and columns, and initialize concentration array
    bands, rows, cols = data_cube.shape
    concentration = np.zeros((rows, cols))

    # Compute the background spectrum (mean across spatial dimensions)
    background_spectrum = np.nanmean(data_cube, axis=(1, 2))

    # Initialize target spectrum with the first unit absorption spectrum
    target_spectrum = background_spectrum * unit_absorption_spectra[0]
    radiancediff_with_bg = data_cube - background_spectrum[:, None, None]

    # Efficient covariance matrix calculation using vectorization
    d_covariance = radiancediff_with_bg.reshape(bands, -1)
    covariance = np.einsum("ij,ik->jk", d_covariance, d_covariance)
    covariance /= rows * cols
    covariance_inverse = np.linalg.pinv(covariance)

    # Adjust albedo if requested
    albedo = np.ones((rows, cols))
    if albedoadjust:
        albedo = np.einsum("ijk,i->jk", data_cube, background_spectrum) / (
            background_spectrum.T @ background_spectrum
        )

    # Vectorized calculation of methane concentration enhancement
    numerator = np.einsum(
        "ijk,jk->ik", radiancediff_with_bg, covariance_inverse @ target_spectrum
    )
    denominator = albedo * (target_spectrum.T @ covariance_inverse @ target_spectrum)
    concentration = numerator / denominator

    # Backup the original concentration
    original_concentration = concentration.copy()

    # Multi-layer absorption spectrum calculation for high-concentration areas
    levelon = True
    adaptive_threshold = 6000
    i = 1

    while levelon:
        # Create masks for high and low concentration areas
        high_concentration_mask = original_concentration > adaptive_threshold * (
            0.99**i
        )
        low_concentration_mask = original_concentration <= adaptive_threshold * (
            0.99**i
        )

        # Proceed with recalculating the high-concentration areas
        if np.sum(high_concentration_mask) > 0 and adaptive_threshold < 32000:
            # Update background spectrum for high concentration areas
            background_spectrum_high = np.nanmean(
                data_cube[:, high_concentration_mask]
                - concentration[high_concentration_mask] * target_spectrum[:, None],
                axis=1,
            )

            # Recalculate the target spectrum using the updated background and absorption spectra
            high_target_spectrum = background_spectrum_high * unit_absorption_spectra[i]

            # Update radiance difference for high and low concentration areas
            radiancediff_with_bg[:, high_concentration_mask] = (
                data_cube[:, high_concentration_mask]
                - background_spectrum_high[:, None]
            )
            radiancediff_with_bg[:, low_concentration_mask] = (
                data_cube[:, low_concentration_mask] - background_spectrum[:, None]
            )

            # Vectorized concentration update for high concentration areas
            numerator_high = np.einsum(
                "ijk,jk->ik",
                radiancediff_with_bg[:, high_concentration_mask],
                high_target_spectrum,
            )
            denominator_high = high_target_spectrum.T @ high_target_spectrum

            concentration[high_concentration_mask] = (
                numerator_high / denominator_high
            ) + adaptive_threshold

            # Update masks for next iteration
            adaptive_threshold += 6000
            i += 1

        else:
            levelon = False

    return original_concentration, concentration


# columnwise multi-layer matched filter algorithm 逐列计算
def columnwise_ml_matched_filter(
    data_array: np.ndarray,
    stacked_unit_absorption_spectrum: np.ndarray,
    is_iterate: bool = False,
    is_albedo: bool = False,
    is_filter: bool = False,
    is_columnwise: bool = False,
) -> np.ndarray:
    """
    Calculate the methane enhancement of the image data based on the original matched filter
    and the unit absorption spectrum.

    :param data_array: numpy array of the image data
    :param unit_absorption_spectrum: list of the unit absorption spectrum
    :param is_iterate: flag to decide whether to iterate the matched filter
    :param is_albedo: flag to decide whether to do the albedo correction
    :param is_filter: flag to decide whether to add the l1-filter correction
    :return: numpy array of methane enhancement result
    """
    # 获取 以 波段 行数 列数 为顺序的数据
    bands, rows, cols = data_array.shape
    # 初始化 concentration 数组，大小与卫星数据尺寸一直
    concentration = np.zeros((rows, cols))
    # 遍历不同列数，目的是为了消除 不同传感器之间带来的误差
    if is_columnwise:
        for col_index in range(cols):
            # 获取当前列的数据
            current_column = data_array[:, :, col_index]
            # 获取当前列的非空行的 索引 以及 数目
            valid_rows = ~np.isnan(current_column[0, :])
            count_not_nan = np.count_nonzero(valid_rows)
            # 对于全为空的列，直接将浓度值设为 nan
            if count_not_nan == 0:
                concentration[:, col_index] = np.nan
                continue

            # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
            background_spectrum = np.nanmean(current_column, axis=1)
            target_spectrum = (
                background_spectrum * stacked_unit_absorption_spectrum[0, :]
            )

            # 对当前目标光谱的每一行进行去均值操作，得到调整后的光谱，以此为基础计算协方差矩阵，并获得其逆矩阵
            radiancediff_with_bg = (
                current_column[:, valid_rows] - background_spectrum[:, None]
            )
            covariance = np.zeros((bands, bands))
            for i in range(count_not_nan):
                covariance += np.outer(
                    radiancediff_with_bg[:, i], radiancediff_with_bg[:, i]
                )
            covariance = covariance / count_not_nan
            covariance_inverse = np.linalg.inv(covariance)

            # 判断是否进行反照率校正，若是，则通过背景光谱和实际光谱计算反照率校正因子
            albedo = np.ones((rows, cols))
            if is_albedo:
                albedo[valid_rows, col_index] = (
                    current_column[:, valid_rows].T @ background_spectrum
                ) / (background_spectrum.T @ background_spectrum)

            # 基于最优化公式计算每个像素的甲烷浓度增强值
            up = radiancediff_with_bg.T @ covariance_inverse @ target_spectrum
            down = albedo[valid_rows, col_index] * (
                target_spectrum.T @ covariance_inverse @ target_spectrum
            )
            concentration[valid_rows, col_index] = up / down

            levelon = True
            # 计算浓度增强值的均值和标准差
            mean_concentration = np.nanmean(
                concentration[valid_rows, col_index]
            )  # 忽略 NaN 值
            std_concentration = np.nanstd(
                concentration[valid_rows, col_index]
            )  # 忽略 NaN 值
            # 使用均值加一个标准差作为自适应阈值
            adaptive_threshold = mean_concentration + std_concentration
            while levelon:
                high_concentration_mask = (
                    concentration[valid_rows, col_index] > adaptive_threshold
                )
                # 获取这个阈值的单位吸收谱，可以通过插值查找表获得
                # 使用新的单位吸收谱重新计算目标光谱
                background_spectrum = np.nanmean(
                    current_column[:, valid_rows]
                    + albedo[valid_rows, col_index]
                    * concentration[valid_rows, col_index]
                    * target_spectrum[:, np.newaxis],
                    axis=1,
                )
                background_spectrum = (
                    background_spectrum
                    + adaptive_threshold * stacked_unit_absorption_spectrum[1, :]
                )
                target_spectrum = np.multiply(
                    background_spectrum, stacked_unit_absorption_spectrum[1, :]
                )
                radiancediff_with_bg = (
                    current_column[:, valid_rows]
                    - background_spectrum[:, None]
                    - albedo[valid_rows, col_index]
                    * (concentration[valid_rows, col_index] - adaptive_threshold)
                    * target_spectrum[:, np.newaxis]
                )
                covariance = np.zeros((bands, bands))
                for i in range(valid_rows.shape[0]):
                    covariance += np.outer(
                        radiancediff_with_bg[:, i], radiancediff_with_bg[:, i]
                    )
                covariance = covariance / count_not_nan
                covariance_inverse = np.linalg.inv(covariance)
                # 基于新的目标光谱重新计算高浓度像素的甲烷浓度增强值
                up = (
                    radiancediff_with_bg[:, high_concentration_mask].T
                    @ covariance_inverse
                    @ target_spectrum
                )
                down = albedo[valid_rows, col_index][high_concentration_mask] * (
                    target_spectrum.T @ covariance_inverse @ target_spectrum
                )
                # 直接更新原数组
                valid_indices = np.where(valid_rows)[0]
                high_concentration_indices = valid_indices[high_concentration_mask]
                concentration[high_concentration_indices, col_index] = (
                    up / down + adaptive_threshold
                )
                # 计算浓度增强值的均值和标准差
                mean_concentration = np.nanmean(
                    concentration[valid_rows, col_index]
                )  # 忽略 NaN 值
                std_concentration = np.nanstd(
                    concentration[valid_rows, col_index]
                )  # 忽略 NaN 值
                # 使用均值加一个标准差作为自适应阈值
                new_adaptive_threshold = mean_concentration + std_concentration
                if (
                    np.abs(
                        (new_adaptive_threshold - adaptive_threshold)
                        / adaptive_threshold
                    )
                    < 0.1
                ):
                    adaptive_threshold = new_adaptive_threshold
                else:
                    levelon = False

            # 判断是否进行迭代，若是，则进行如下迭代计算
            if is_iterate:
                l1filter = np.zeros((rows, cols))
                epsilon = np.finfo(np.float32).tiny
                for iter_num in range(5):
                    if is_filter:
                        l1filter[valid_rows, col_index] = 1 / (
                            concentration[valid_rows, col_index] + epsilon
                        )
                    else:
                        l1filter[valid_rows, col_index] = 0

                    # 更新背景光谱和目标光谱
                    column_replacement = (
                        current_column[:, valid_rows]
                        - (
                            albedo[valid_rows, col_index]
                            * concentration[valid_rows, col_index]
                        )[None, :]
                        * target_spectrum[:, None]
                    )
                    # 计算更新后的 背景光谱 和 目标谱
                    background_spectrum = np.mean(column_replacement, axis=1)
                    target_spectrum = np.multiply(
                        background_spectrum, stacked_unit_absorption_spectrum[0, :]
                    )
                    # 基于新的目标谱 和 背景光谱 计算协方差矩阵
                    radiancediff_with_bg = (
                        current_column[:, valid_rows]
                        - (
                            albedo[valid_rows, col_index]
                            * concentration[valid_rows, col_index]
                        )[None, :]
                        * target_spectrum[:, None]
                        - background_spectrum[:, None]
                    )
                    covariance = np.zeros((bands, bands))
                    for i in range(valid_rows.shape[0]):
                        covariance += np.outer(
                            radiancediff_with_bg[:, i], radiancediff_with_bg[:, i]
                        )
                    covariance = covariance / count_not_nan
                    covariance_inverse = np.linalg.inv(covariance)

                    # 计算新的甲烷浓度增强值
                    up = (
                        radiancediff_with_bg.T @ covariance_inverse @ target_spectrum
                    ) - l1filter[valid_rows, col_index]
                    down = albedo[valid_rows, col_index] * (
                        target_spectrum.T @ covariance_inverse @ target_spectrum
                    )
                    concentration[valid_rows, col_index] = np.maximum(up / down, 0.0)
                    high_concentration_mask = (
                        concentration[valid_rows, col_index] > 5000
                    )

                    if np.any(high_concentration_mask):
                        # 使用新的单位吸收谱重新计算目标光谱
                        con = concentration[valid_rows, col_index].copy()
                        background_spectrum = np.nanmean(
                            current_column[:, valid_rows]
                            - albedo[valid_rows, col_index]
                            * con
                            * target_spectrum[:, np.newaxis],
                            axis=1,
                        )
                        target_spectrum = np.multiply(
                            background_spectrum, stacked_unit_absorption_spectrum
                        )
                        radiancediff_with_bg = (
                            current_column[:, valid_rows]
                            - albedo[valid_rows, col_index]
                            * con
                            * target_spectrum[:, np.newaxis]
                            - background_spectrum[:, None]
                        )
                        covariance = np.zeros((bands, bands))
                        for i in range(valid_rows.shape[0]):
                            covariance += np.outer(
                                radiancediff_with_bg[:, i], radiancediff_with_bg[:, i]
                            )
                        covariance = covariance / count_not_nan
                        covariance_inverse = np.linalg.inv(covariance)
                        # 基于新的目标光谱重新计算高浓度像素的甲烷浓度增强值
                        up = (
                            radiancediff_with_bg[:, high_concentration_mask].T
                            @ covariance_inverse
                            @ target_spectrum
                        )
                        down = albedo[valid_rows, col_index][
                            high_concentration_mask
                        ] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                        # 直接更新原数组
                        valid_indices = np.where(valid_rows)[0]
                        high_concentration_indices = valid_indices[
                            high_concentration_mask
                        ]
                        concentration[high_concentration_indices, col_index] = (
                            up / down + 2500
                        )

    if not is_columnwise:
        count_not_nan = np.count_nonzero(~np.isnan(data_array[0, :, :]))
        background_spectrum = np.nanmean(data_array, axis=(1, 2))
        target_spectrum = np.multiply(
            background_spectrum, stacked_unit_absorption_spectrum[0, :]
        )
        radiancediff_with_bg = data_array - background_spectrum[:, None, None]
        covariance = np.zeros((bands, bands))
        for i in range(rows):
            for j in range(cols):
                covariance = covariance + np.outer(
                    radiancediff_with_bg[:, i, j], radiancediff_with_bg[:, i, j]
                )
        covariance = covariance / count_not_nan
        covariance_inverse = np.linalg.inv(covariance)
        albedo = np.ones((rows, cols))
        for row_index in range(rows):
            for col_index in range(cols):
                if is_albedo:
                    albedo[row_index, col_index] = (
                        data_array[:, row_index, col_index].T @ background_spectrum
                    ) / (background_spectrum.T @ background_spectrum)
                up = (
                    radiancediff_with_bg[:, row_index, col_index].T
                    @ covariance_inverse
                    @ target_spectrum
                )
                down = albedo[row_index, col_index] * (
                    target_spectrum.T @ covariance_inverse @ target_spectrum
                )
                concentration[row_index, col_index] = up / down

        if is_iterate:
            l1filter = np.zeros((rows, cols))
            epsilon = np.finfo(np.float32).tiny
            iter_data = data_array.copy()

            for iter_num in range(5):
                if is_filter:
                    l1filter = 1 / (concentration + epsilon)
                iter_data = data_array - (
                    target_spectrum[:, None, None]
                    * albedo[None, :, :]
                    * concentration[None, :, :]
                )
                background_spectrum = np.nanmean(iter_data, axis=(1, 2))
                target_spectrum = np.multiply(
                    background_spectrum, stacked_unit_absorption_spectrum[0, :]
                )
                radiancediff_with_bg = data_array - background_spectrum[:, None, None]
                covariance = np.zeros((bands, bands))
                for i in range(rows):
                    for j in range(cols):
                        covariance += np.outer(
                            radiancediff_with_bg[:, i, j], radiancediff_with_bg[:, i, j]
                        )
                covariance = covariance / count_not_nan
                covariance_inverse = np.linalg.inv(covariance)

                for row_index in range(rows):
                    for col_index in range(cols):
                        up = (
                            radiancediff_with_bg[:, row_index, col_index].T
                            @ covariance_inverse
                            @ target_spectrum
                        )
                        down = albedo[row_index, col_index] * (
                            target_spectrum.T @ covariance_inverse @ target_spectrum
                        )
                        concentration[row_index, col_index] = np.maximum(up / down, 0)

    # 返回 甲烷浓度增强的结果
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
    d_covariance = np.log(data_cube) - background_spectrum[:, None, None]
    covariance = np.zeros((bands, bands))
    for row in range(rows):
        for col in range(cols):
            covariance += np.outer(d_covariance[:, row, col], d_covariance[:, row, col])
    covariance = covariance / (rows * cols)
    covariance_inverse = np.linalg.inv(covariance)

    general_denominator = (
        unit_absorption_spectrum.T @ covariance_inverse @ unit_absorption_spectrum
    )

    for row in range(rows):
        for col in range(cols):
            # 基于最优化公式计算每个像素的甲烷浓度增强值
            numerator = (
                radiancediff_with_bg[:, row, col].T
                @ covariance_inverse
                @ unit_absorption_spectrum
            )
            concentration[row, col] = numerator / general_denominator
    return concentration


# convert the radiance into log space 整幅图像进行计算
def columnwise_lognormal_matched_filter(
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
    d_covariance = np.log(data_cube) - background_spectrum[:, None, None]
    covariance = np.zeros((bands, bands))
    for row in range(rows):
        for col in range(cols):
            covariance += np.outer(d_covariance[:, row, col], d_covariance[:, row, col])
    covariance = covariance / (rows * cols)
    covariance_inverse = np.linalg.inv(covariance)

    general_denominator = (
        unit_absorption_spectrum.T @ covariance_inverse @ unit_absorption_spectrum
    )

    for row in range(rows):
        for col in range(cols):
            # 基于最优化公式计算每个像素的甲烷浓度增强值
            numerator = (
                radiancediff_with_bg[:, row, col].T
                @ covariance_inverse
                @ unit_absorption_spectrum
            )
            concentration[row, col] = numerator / general_denominator
    return concentration


# Kalman filter and matched filter 整幅图像进行计算
def Kalman_filterr_matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    albedoadjust: bool,
    iterate: bool,
    sparsity: bool,
):
    return None


def columnwise_Kalman_filterr_matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    albedoadjust: bool,
    iterate: bool,
    sparsity: bool,
):
    return None


# 对算法进行验证
def algorithm_validation():
    image_sample = np.random.rand(242, 242, 242)
    unit_absorption_spectrum = np.random.rand(242)
    albedoadjust = True
    iterate = True
    sparsity = True
    result = matched_filter(
        image_sample, unit_absorption_spectrum, albedoadjust, iterate, sparsity
    )

    return result


if __name__ == "__main__":
    algorithm_validation()
    print("algorithm validation is done.")
