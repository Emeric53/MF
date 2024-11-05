import numpy as np


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
            # radiancediff_with_background[:, low_concentration_mask] = (
            #     data_cube[:, low_concentration_mask] - background_spectrum[:, None]
            # )

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
            # low_concentration_mask = original_concentration <= adaptive_threshold * 0.99

            adaptive_threshold += 6000
            i += 1

        else:
            levelon = False

    return original_concentration, concentration


if __name__ == "__main__":
    data_cube = np.random.rand(10, 100, 100)
    unit_absorption_spectrum_array = np.random.rand(10, 10)
    albedoadjust = True
    ml_matched_filter(data_cube, unit_absorption_spectrum_array, albedoadjust)
