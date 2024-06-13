import numpy as np


def open_unit_absorption_spectrum(filepath: str) -> np.array:
    """
    Open the unit absorption spectrum file, filter it by the spectrum range, and convert it to a NumPy array.

    :param filepath: Path to the unit absorption spectrum file
    :return: NumPy array of the filtered unit absorption spectrum
    """
    try:
        with open(filepath, 'r') as file:
            # 读取文件并过滤波长范围
            uas_list = [
                [float(line.split(' ')[0]), float(line.split(' ')[1].strip())]
                for line in file.readlines()
            ]

        # 转换为 NumPy 数组并返回
        return np.array(uas_list)

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def filter_and_slice(arr: np.array, min_val: float, max_val: float) -> (np.array, slice):
    """
    根据最大最小值阈值，筛选数组并获取原数组的切片。

    :param arr: 输入的 NumPy 数组
    :param min_val: 最小值阈值
    :param max_val: 最大值阈值
    :return: 筛选后的数组和原数组中的切片
    """
    # 筛选条件
    condition = (arr >= min_val) & (arr <= max_val)

    # 筛选后的数组
    filtered_arr = arr[condition]

    # 计算切片范围
    nonzero_indices = np.nonzero(condition)[0]
    if len(nonzero_indices) == 0:
        return filtered_arr, None  # 如果没有满足条件的元素，返回None

    slice_start = nonzero_indices[0]
    slice_end = nonzero_indices[-1] + 1  # +1 因为 slice 的结束索引是开区间

    # 创建切片
    arr_slice = slice(slice_start, slice_end)

    return filtered_arr, arr_slice


def matched_filter(data_array: np.array, unit_absorption_spectrum: np.array, is_iterate=False,
                   is_albedo=False, is_filter=False) -> np.array:
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
    # 初始化 albedo 和 concentration 数组，大小与卫星数据尺寸一直
    albedo = np.ones((rows, cols))
    concentration = np.zeros((rows, cols))
    # 遍历不同列数，目的是为了消除 不同传感器之间带来的误差
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

        # 对于非空列，去均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
        background_spectrum = np.nanmean(current_column, axis=1)
        target_spectrum = np.multiply(background_spectrum, unit_absorption_spectrum)

        # 对当前目标光谱的每一行进行去均值操作，得到调整后的光谱，以此为基础计算协方差矩阵，并获得其逆矩阵
        adjusted_column = current_column[:, valid_rows] - background_spectrum[:, None]
        covariance = np.cov(adjusted_column, rowvar=True)/count_not_nan
        covariance_inverse = np.linalg.inv(covariance)

        # 判断是否进行反照率校正，若是，则通过背景光谱和实际光谱计算反照率校正因子
        if is_albedo:
            albedo[valid_rows, col_index] = (
                    (current_column[:, valid_rows].T @ background_spectrum) /
                    (background_spectrum.T @ background_spectrum)
            )

        # 基于最优化公式计算每个像素的甲烷浓度增强值
        up = (adjusted_column.T @ covariance_inverse @ target_spectrum)
        down = albedo[valid_rows, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
        concentration[valid_rows, col_index] = up / down

        # 判断是否进行迭代，若是，则进行如下迭代计算
        if is_iterate:
            l1filter = np.zeros((rows, cols))
            epsilon = np.finfo(np.float32).tiny
            iter_data = current_column.copy()

            for iter_num in range(1):

                if is_filter:
                    l1filter[valid_rows, col_index] = 1 / (concentration[valid_rows, col_index] + epsilon)
                else:
                    l1filter[valid_rows, col_index] = 0
                # 更新背景光谱和目标光谱
                iter_data[:, valid_rows] = current_column[:, valid_rows] - (
                        target_spectrum[:, None] * albedo[valid_rows, col_index][:, None].T *
                        concentration[valid_rows, col_index][:, None].T
                )
                # 计算更新后的背景光谱 和 目标谱
                background_spectrum = np.nanmean(iter_data, axis=1)
                target_spectrum = np.multiply(background_spectrum, unit_absorption_spectrum)
                background_spectrum = background_spectrum[:, np.newaxis]
                # 基于新的目标谱 和 背景光谱 计算协方差矩阵
                adjusted_column = current_column[:, valid_rows] - (
                        target_spectrum[:, None] * albedo[valid_rows, col_index][:, None].T *
                        concentration[valid_rows, col_index][:, None].T
                ) - background_spectrum
                covariance = np.cov(adjusted_column, rowvar=True)
                covariance_inverse = np.linalg.inv(covariance)

                # 计算新的甲烷浓度增强值
                up = (adjusted_column.T @ covariance_inverse @ target_spectrum) - l1filter[valid_rows, col_index]
                down = albedo[valid_rows, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                concentration[valid_rows, col_index] = np.maximum(up / down, 0)

    # 返回 甲烷浓度增强的结果
    return concentration


def main():
    # define the path of the unit absorption spectrum file and open it
    uas_filepath = r'C:\Users\RS\PycharmProjects\MF\unit_absorption_spectrum.txt'
    uas = open_unit_absorption_spectrum(uas_filepath)
    print(uas)

if __name__ == '__main__':
    main()
