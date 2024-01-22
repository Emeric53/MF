from scipy.signal import convolve
import numpy as np
from osgeo import gdal
import sys
import math
# 设置初始栅格文件路径
file_path = "C:\\Users\\RS\\Desktop\\EMIT\\Imageswithplume\\tiff\\EMIT_L1B_RAD_001_20230204T041009_2303503_016_subset.tif"
delta_crosssection = 5.251440616846936e-22
# 利用gdal打开数据
dataset = gdal.Open(file_path, gdal.GA_ReadOnly)

# 从已存在的TIFF文件中获取地理参考信息
geo_transform = dataset.GetGeoTransform()
projection = dataset.GetProjection()

# 获取所有波段数目
num_bands = dataset.RasterCount

# 定义数组存储各波段数据,以及波段中非nan的相关个数
band_data_list = []

# 遍历各个波段，注意数据集的getrasterband方法索引从1开始
for band_index in range(1, num_bands + 1):
    # 依据索引获取当前波段数据
    current_band = dataset.GetRasterBand(band_index)
    # 读取波段数据为NumPy数组
    current_band_data = current_band.ReadAsArray()
    # 添加入波段数据总和中
    band_data_list.append(current_band_data)

# 将波段数据转为np数组array
image_data = np.array(band_data_list)

# 获取栅格数据的波段，行，列
bands, rows, cols = image_data.shape
b1 = image_data[37, :, :]
b2 = image_data[38, :, :]
concentration = np.zeros((rows, cols))
for row in range(rows):
    for col in range(cols):
        result = math.log(b2[row, col] / b1[row, col], math.e) / delta_crosssection
        concentration[row, col] = result
    # 指定输出的TIFF文件名
output_tiff_file = "C:\\Users\\RS\\Desktop\\emit_test.tiff"
# 获取数组的维度
rows, cols = concentration.shape

# 创建一个新的TIFF文件
driver = gdal.GetDriverByName('GTiff')
dataset = driver.Create(output_tiff_file, cols, rows, 1, gdal.GDT_Float32)

# 将NumPy数组写入TIFF文件
band = dataset.GetRasterBand(1)
band.WriteArray(concentration)

# 设置获取的地理参考信息
dataset.SetGeoTransform(geo_transform)
dataset.SetProjection(projection)
