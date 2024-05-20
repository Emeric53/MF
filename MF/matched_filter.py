import numpy as np


def open_unit_absorption_spectrum(filepath, min_wavelength, max_wavelength):
    """
    Open the unit absorption spectrum file, filter it by the spectrum range, and convert it to a NumPy array.

    :param filepath: Path to the unit absorption spectrum file
    :param min_wavelength: Minimum wavelength for filtering
    :param max_wavelength: Maximum wavelength for filtering
    :return: NumPy array of the filtered unit absorption spectrum
    """
    try:
        with open(filepath, 'r') as file:
            # 读取文件并过滤波长范围
            uas_list = [
                float(line.split(' ')[1].strip())
                for line in file.readlines()
                if min_wavelength <= float(line.split(' ')[0]) <= max_wavelength
            ]

        # 转换为 NumPy 数组并返回
        return np.array(uas_list)

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def matched_filter(data_array, unit_absorption_spectrum, is_iterate=False, is_albedo=False, is_filter=False):
    """
    Calculate the methane enhancement of the image data based on the original matched filter and the unit absorption spectrum.

    :param data_array: numpy array of the image data
    :param unit_absorption_spectrum: list of the unit absorption spectrum
    :param is_iterate: flag to decide whether to iterate the matched filter
    :param is_albedo: flag to decide whether to do the albedo correction
    :param is_filter: flag to decide whether to add the l1-filter correction
    :return: numpy array of methane enhancement result
    """
    rows, cols, bands = data_array.shape
    albedo = np.ones((rows, cols))
    concentration = np.zeros((rows, cols))

    for col_index in range(cols):
        current_column = data_array[:, col_index, :]
        valid_rows = ~np.isnan(current_column[:, 0])
        count_not_nan = np.count_nonzero(valid_rows)
        if count_not_nan == 0:
            concentration[:, col_index] = np.nan
            continue

        background_spectrum = np.nanmean(current_column, axis=0)
        target_spectrum = np.multiply(background_spectrum, unit_absorption_spectrum)
        adjusted_column = current_column[valid_rows, :] - background_spectrum
        covariance = np.einsum('ij,ik->jk', adjusted_column, adjusted_column) / count_not_nan
        covariance_inverse = np.linalg.inv(covariance)

        if is_albedo:
            albedo[valid_rows, col_index] = (
                np.einsum('ij,j->i', current_column[valid_rows, :], background_spectrum) /
                np.inner(background_spectrum, background_spectrum)
            )

        up = np.einsum('ij,jk,k->i', current_column[valid_rows, :] - background_spectrum,
                       covariance_inverse, target_spectrum)
        down = albedo[valid_rows, col_index] * (target_spectrum @ covariance_inverse @ target_spectrum)
        concentration[valid_rows, col_index] = up / down

        if is_iterate:
            l1filter = np.zeros((rows, cols))
            epsilon = np.finfo(np.float32).tiny
            iter_data = current_column.copy()

            for iter_num in range(4):
                iter_data[valid_rows, :] = current_column[valid_rows, :] - (
                    albedo[valid_rows, col_index][:, None] *
                    target_spectrum *
                    concentration[valid_rows, col_index][:, None]
                )

                if is_filter:
                    l1filter[valid_rows, col_index] = 1 / (concentration[valid_rows, col_index] + epsilon)
                else:
                    l1filter[valid_rows, col_index] = 0

                background_spectrum = np.nanmean(iter_data, axis=0)
                target_spectrum = np.multiply(background_spectrum, unit_absorption_spectrum)
                diffs = iter_data[valid_rows, :] - background_spectrum
                covariance = np.einsum('ij,ik->jk', diffs, diffs) / count_not_nan
                covariance_inverse = np.linalg.inv(covariance)

                up = np.einsum('ij,jk,k->i', current_column[valid_rows, :] - background_spectrum,
                               covariance_inverse, target_spectrum) - l1filter[valid_rows, col_index]
                down = albedo[valid_rows, col_index] * (target_spectrum @ covariance_inverse @ target_spectrum)
                concentration[valid_rows, col_index] = np.maximum(up / down, 0)

    return concentration

