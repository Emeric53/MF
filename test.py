import numpy as np
import math


def rad_calibration(dataset, cal_file="GF5B_AHSI_RadCal_SWIR.raw"):
    """
    Perform radiation calibration on the AHSI L1 data using calibration coefficients.

    :param dataset: 3D NumPy array of shape (bands, height, width)
    :param cal_file: Path to the calibration file containing coefficients
    :return: Calibrated dataset as a 3D NumPy array
    """
    try:
        # 读取校准文件
        with open(cal_file, "r") as file:
            lines = file.readlines()

        # 提取校准系数
        coeffs = [tuple(map(float, line.split(','))) for line in lines]

        # 检查数据集的波段数是否与校准系数的数量匹配
        if len(coeffs) != dataset.shape[0]:
            raise ValueError("The number of calibration coefficients does not match the number of bands in the dataset.")

        # 应用校准系数
        for index, (a, b) in enumerate(coeffs):
            dataset[index, :, :] = a * dataset[index, :, :] + b

        return dataset

    except FileNotFoundError:
        print(f"Error: The calibration file '{cal_file}' was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

