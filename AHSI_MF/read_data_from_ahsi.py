import numpy as np
from osgeo import gdal
import os

from osgeo import gdal
import numpy as np


def get_raster_array(filepath):
    """
    Reads a raster file and returns a NumPy array containing all the bands.

    :param filepath: the path of the raster file
    :return: a 3D NumPy array with shape (bands, height, width)
    """
    try:
        # 打开文件路径中的数据集
        dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(f"Unable to open file: {filepath}")
    except Exception as e:
        print(f"Error: {e}")
        return None

    # 获取波段数
    band_count = dataset.RasterCount

    # 创建一个 NumPy 数组来存储所有波段的数据
    data_array = np.array([dataset.GetRasterBand(i).ReadAsArray() for i in range(1, band_count + 1)])

    return data_array







# operate the radiation calibration on the AHSI l1 data
def rad_calibration(dataset):
    with open("GF5B_AHSI_RadCal_SWIR.raw", "r") as file:
        lines = file.readlines()
        index = 0
        for line in lines:
            a = float(line.split(',')[0])
            b = float(line.split(",")[1].rstrip('\n'))
            dataset[index, :, :] = a * dataset[index, :, :] + b
            index += 1
    return dataset


# open the unit absorption spectrum and filter with the spectrum range
def open_unit_absorption_spectrum(filepath, min, max):
    # open the unit absorption spectrum file and convert it a numpy array
    uas_list = []
    with open(filepath, 'r') as file:
        data = file.readlines()
        for band in data:
            split_i = band.split(' ')
            wvl = float(split_i[0])
            if min <= wvl <= max:
                band = split_i[1].rstrip('\n')
                uas_list.append(float(band))
    out_put = np.array(uas_list)
    return out_put