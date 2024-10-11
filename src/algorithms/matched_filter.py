import numpy as np
from concurrent.futures import ThreadPoolExecutor


def compute_background_and_target_spectrum_and_radiance_diff_and_d_covariance(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    polluted: np.ndarray = None,
):
    """
    Compute the background spectrum and target spectrum.
    """
    if polluted:
        if data_cube.ndim == 3:
            background_spectrum = np.nanmean(data_cube - polluted, axis=(1, 2))
        elif data_cube.ndim == 2:
            background_spectrum = np.nanmean(data_cube - polluted, axis=1)
        else:
            raise ValueError("Data cube must be 2D or 3D")
        target_spectrum = background_spectrum * unit_absorption_spectrum
        data_cube_diff = data_cube - background_spectrum[:, None, None]
        d_covariance = data_cube_diff - polluted
    else:
        if data_cube.ndim == 3:
            background_spectrum = np.nanmean(data_cube, axis=(1, 2))
        elif data_cube.ndim == 2:
            background_spectrum = np.nanmean(data_cube, axis=1)
        else:
            raise ValueError("Data cube must be 2D or 3D")
        target_spectrum = background_spectrum * unit_absorption_spectrum
        data_cube_diff = data_cube - polluted - background_spectrum[:, None, None]
        d_covariance = data_cube_diff
    return background_spectrum, target_spectrum, data_cube_diff, d_covariance


def compute_covariance_inverse(
    radiancediff_with_background: np.ndarray,
    counts: int,
):
    """
    Compute the covariance matrix and its inverse.
    """
    covariance = (
        np.tensordot(
            radiancediff_with_background,
            radiancediff_with_background,
            axes=((1, 2), (1, 2)),
        )
        / counts
    )
    return np.linalg.pinv(covariance)


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
    l1filter: float,
    common_denominator: float,
):
    """
    Compute concentration for a single pixel.
    """
    numerator = radiancediff_with_background.T @ covariance_inverse @ target_spectrum
    return numerator - l1filter / (albedo * common_denominator)


def compute_concentration_column(
    column_radiancediff_with_bg: np.ndarray,
    covariance_inverse: np.ndarray,
    target_spectrum: np.ndarray,
    column_albedo: np.ndarray,
    common_denominator: float,
):
    # Initial concentration calculation
    numerator = column_radiancediff_with_bg.T @ covariance_inverse @ target_spectrum
    denominator = column_albedo * common_denominator
    column_concentration = numerator / denominator
    return column_concentration


def matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    iterate: bool,
    albedoadjust: bool,
    sparsity: bool,
) -> np.ndarray:
    """
    Calculate methane enhancement based on the original matched filter method.
    """
    _, rows, cols = data_cube.shape
    concentration = np.zeros((rows, cols))
    albedo = np.ones((rows, cols))
    l1filter = np.zeros((rows, cols))
    # Compute initial background and target spectrum, radiance difference with background, and covariance inverse
    background_spectrum, target_spectrum, radiancediff_with_background, d_covariance = (
        compute_background_and_target_spectrum_and_radiance_diff_and_d_covariance(
            data_cube, unit_absorption_spectrum
        )
    )
    covariance_inverse = compute_covariance_inverse(d_covariance, rows * cols)

    # Compute albedo adjustment if required
    if albedoadjust:
        albedo = compute_albedo(data_cube, background_spectrum)

    # Compute common denominator
    common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum

    # Compute concentration for each pixel
    for row in range(rows):
        for col in range(cols):
            concentration[row, col] = compute_concentration_pixel(
                radiancediff_with_background[:, row, col],
                covariance_inverse,
                target_spectrum,
                albedo[row, col],
                l1filter[row, col],
                common_denominator,
            )

    # Perform iterative updates if requested
    if iterate:
        for _ in range(5):
            if sparsity:
                l1filter = 1 / (concentration + np.finfo(np.float64).tiny)

            # Update background, target spectrum, radiance difference with background, and covariance inverse
            (
                background_spectrum,
                target_spectrum,
                radiancediff_with_background,
                d_covariance,
            ) = compute_background_and_target_spectrum_and_radiance_diff_and_d_covariance(
                data_cube,
                unit_absorption_spectrum,
                albedo[:, :, None] * concentration[:, :, None] * target_spectrum,
            )

            covariance_inverse = compute_covariance_inverse(d_covariance, rows, cols)

            common_denominator = (
                target_spectrum.T @ covariance_inverse @ target_spectrum
            )

            for row in range(rows):
                for col in range(cols):
                    concentration[row, col] = np.max(
                        compute_concentration_pixel(
                            radiancediff_with_background[:, row, col],
                            covariance_inverse,
                            target_spectrum,
                            albedo[row, col],
                            l1filter[row, col],
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
    sparsity: bool = False,
) -> np.ndarray:
    """
    Perform matched filter calculation column by column using multithreading.
    """

    def process_column(
        data_cube: np.ndarray,
        unit_absorption_spectrum: np.ndarray,
        col_index: int,
        iterate: bool,
        albedoadjust: bool,
        sparsity: bool,
    ):
        """
        Process a single column of the data cube.
        """
        _, rows, _ = data_cube.shape
        current_column = data_cube[:, :, col_index]
        valid_rows = ~np.isnan(current_column[0, :])
        count_not_nan = np.count_nonzero(valid_rows)

        if count_not_nan == 0:
            return np.nan * np.zeros(rows)

        # Compute background and target spectrum and radiance difference with background and covariance inverse
        background_spectrum, target_spectrum, radiancediff_with_bg, d_covariance = (
            compute_background_and_target_spectrum_and_radiance_diff_and_d_covariance(
                current_column, unit_absorption_spectrum
            )
        )
        covariance_inverse = compute_covariance_inverse(d_covariance, count_not_nan)

        if albedoadjust:
            albedo = compute_albedo(current_column, background_spectrum)

        common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum
        concentration = compute_concentration_column(
            radiancediff_with_bg,
            covariance_inverse,
            target_spectrum,
            albedo,
            common_denominator,
        )
        # Iterative updates
        if iterate:
            l1filter = np.zeros(rows)
            epsilon = np.finfo(np.float64).tiny
            for _ in range(5):
                if sparsity:
                    l1filter[valid_rows] = 1 / (concentration[valid_rows] + epsilon)

                (
                    background_spectrum,
                    target_spectrum,
                    radiancediff_with_bg,
                    d_covariance,
                ) = compute_background_and_target_spectrum_and_radiance_diff_and_d_covariance(
                    current_column[:, valid_rows]
                    - albedo[valid_rows]
                    * concentration[valid_rows]
                    * target_spectrum[:, None],
                    unit_absorption_spectrum,
                )

                covariance_inverse = compute_covariance_inverse(
                    d_covariance, count_not_nan
                )

                common_denominator = (
                    target_spectrum.T @ covariance_inverse @ target_spectrum
                )

                concentration[valid_rows] = compute_concentration_column(
                    radiancediff_with_bg,
                    covariance_inverse,
                    target_spectrum,
                    albedo,
                    common_denominator,
                )

        return concentration

    _, _, cols = data_cube.shape

    # Parallel column processing
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                lambda col_index: process_column(
                    data_cube,
                    unit_absorption_spectrum,
                    col_index,
                    iterate,
                    albedoadjust,
                    sparsity,
                ),
                range(cols),
            )
        )

    concentration = np.stack(results, axis=1)
    return concentration


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

    # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
    background_spectrum, target_spectrum, radiance_diff_with_background = (
        compute_background_and_target_spectrum_and_radiance_diff(
            data_cube, unit_absorption_spectrum_array[0]
        )
    )

    # 计算协方差矩阵，并获得其逆矩阵
    d_covariance = radiance_diff_with_background
    covariance_inverse = compute_covariance_inverse(d_covariance, rows * cols)

    # 判断是否进行反照率校正，若是，则通过背景光谱和实际光谱计算反照率校正因子
    albedo = np.ones((rows, cols))
    if albedoadjust:
        compute_albedo(data_cube, background_spectrum)

    # Compute concentration for each pixel
    common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum
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


def columnwise_ml_matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    iterate: bool = False,
    albedoadjust: bool = False,
    sparsity: bool = False,
) -> np.ndarray:
    """
    Perform matched filter calculation column by column using multithreading.
    """

    def process_column(
        data_cube: np.ndarray,
        unit_absorption_spectrum_array: np.ndarray,
        col_index: int,
        albedoadjust: bool,
    ):
        """
        Process a single column of the data cube.
        """
        _, rows, _ = data_cube.shape
        current_column = data_cube[:, :, col_index]
        valid_rows = ~np.isnan(current_column[0, :])
        count_not_nan = np.count_nonzero(valid_rows)

        if count_not_nan == 0:
            return np.nan * np.zeros(rows)

        # Compute background and target spectrum
        background_spectrum, target_spectrum, radiancediff_with_bg = (
            compute_background_and_target_spectrum_and_radiance_diff(
                current_column, unit_absorption_spectrum
            )
        )

        # Compute covariance inverse
        d_covariance = radiancediff_with_bg
        covariance_inverse = compute_covariance_inverse(d_covariance, count_not_nan)

        # Compute albedo adjustment if required
        albedo = np.ones(rows)

        if albedoadjust:
            albedo = compute_albedo(current_column, background_spectrum)

        common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum
        concentration = compute_concentration_column(
            radiancediff_with_bg,
            covariance_inverse,
            target_spectrum,
            albedo,
            common_denominator,
        )
        # Iterative updates
        if iterate:
            l1filter = np.zeros(rows)
            epsilon = np.finfo(np.float64).tiny
            for _ in range(5):
                if sparsity:
                    l1filter[valid_rows] = 1 / (concentration[valid_rows] + epsilon)

                background_spectrum, target_spectrum, radiancediff_with_bg = (
                    compute_background_and_target_spectrum_and_radiance_diff(
                        current_column[:, valid_rows]
                        - albedo[valid_rows]
                        * concentration[valid_rows]
                        * target_spectrum[:, None],
                        unit_absorption_spectrum,
                    )
                )

                radiancediff_with_bg = (
                    current_column[:, valid_rows] - background_spectrum[:, None]
                )

                d_covariance = (
                    radiancediff_with_bg
                    - albedo[valid_rows]
                    * concentration[valid_rows]
                    * target_spectrum[:, None]
                )
                covariance_inverse = compute_covariance_inverse(
                    d_covariance, count_not_nan
                )

                numerator = (
                    radiancediff_with_bg.T @ covariance_inverse @ target_spectrum
                    - l1filter[valid_rows]
                )
                denominator = albedo[valid_rows] * (
                    target_spectrum.T @ covariance_inverse @ target_spectrum
                )
                concentration[valid_rows] = np.maximum(numerator / denominator, 0.0)

        return concentration

    _, _, cols = data_cube.shape

    # Parallel column processing
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                lambda col_index: process_column(
                    data_cube,
                    unit_absorption_spectrum,
                    col_index,
                    albedoadjust,
                ),
                range(cols),
            )
        )

    concentration = np.stack(results, axis=1)
    return concentration
