"""本程序用于对EMIT radiacne数据 基于匹配滤波算法 进行甲烷浓度增强的反演"""
import numpy as np
from osgeo import gdal

def get_raster_array(filepath):
    # 利用gdal打开数据
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    return dataset

# 定义文件路径
radiance_path = "C:\\Users\\RS\\Desktop\\EMIT\\Radiance\\EMIT_L1B_RAD_001_20230420T060148_2311004_021_subset.tif"

dataset = get_raster_array(radiance_path)
# 从已存在的TIFF文件中获取地理参考信息以及波段信息
geo_transform = dataset.GetGeoTransform()
projection = dataset.GetProjection()
num_bands = dataset.RasterCount

# 定义数组存储各波段数据,以及波段中非nan的相关个数
band_data_list = []
count_non_nan = 0

# 遍历各个波段
for band_index in range(1, num_bands + 1):
    # 依据索引获取波段数据
    band = dataset.GetRasterBand(band_index)
    # 读取波段数据为NumPy数组
    band_data = band.ReadAsArray()
    # 添加入波段数据总和中
    band_data_list.append(band_data)
    # 计算非nan像元的个数
    count_non_nan = np.count_nonzero(~np.isnan(band_data))

# 将波段数据转为array
image_data = np.array(band_data_list)

# 获取栅格数据的波段，行，列
bands, rows, cols = image_data.shape


def open_unit_absorption_spectrum(filepath):
    # 打开AHSI的单位吸收光谱文件并转换为numpy数组
    unitabsorptionspectrum = []
    with open(filepath, 'r') as file:
        data = file.readlines()
        for band in data:
            unitabsorptionspectrum.append(float(band))
    output = np.array(unitabsorptionspectrum)
    return output


# 构造总的甲烷浓度增强  以及用于地表反照率校正的二维数组变量
albedo = np.zeros((rows, cols))
# 构造用于稀疏校正的l1校正项 数组
l1filter = np.zeros((rows, cols))
# 以每个波段为分割，计算非nan的平均值
u = np.nanmean(image_data, axis=(1, 2))

# 构造协方差矩阵
c = np.zeros((bands, bands))
for row in range(rows):
    for col in range(cols):
        if not np.isnan(image_data[0, row, col]):
            c += np.outer(image_data[:, row, col] - u, image_data[:, row, col] - u)
c = c / count_non_nan
c_inverse = np.linalg.inv(c)

uas_filepath = 'EMIT_unit_absorption_spectrum.txt'
unitabsorptionspectrum = open_unit_absorption_spectrum(uas_filepath)
target = np.multiply(u, unitabsorptionspectrum)

# 构造初始的甲烷浓度
alpha = np.zeros((rows, cols))

for row in range(rows):
    for col in range(cols):
        albedo[row, col] = (np.inner(image_data[:, row, col], u)
                            / np.inner(u, u))
        if not np.isnan(image_data[0, row, col]):
            up = (image_data[:, row, col] - u) @ c_inverse @ target
            down = albedo[row, col] * (target @ c_inverse @ target)
            alpha[row, col] = up / down
        else:
            alpha[row, col] = np.nan

# 构造用于稀疏校正的l1校正项 数组
l1filter = np.zeros((rows, cols))

for i in range(20):
    iter_data = image_data.copy()
    for row in range(rows):
        for col in range(cols):
            if not np.isnan(image_data[0, row, col]):
                iter_data[:, row, col] = image_data[:, row, col] - target * alpha[row, col]
    u = np.nanmean(iter_data, axis=(1, 2))
    c = np.zeros((bands, bands))
    target = np.multiply(u, unitabsorptionspectrum)
    count = 0
    for row in range(rows):
        for col in range(cols):
            if not np.isnan(image_data[0, row, col]):
                c += np.outer(image_data[:, row, col] - (u + albedo[row, col] * alpha[row, col] * target),
                              image_data[:, row, col] - (u + albedo[row, col] * alpha[row, col] * target))
    c = c / count_non_nan
    c_inverse = np.linalg.inv(c)
    for row in range(rows):
        for col in range(cols):
            if not np.isnan(image_data[0, row, col]):
                up = (image_data[:, row, col] - u) @ c_inverse @ target
                down = albedo[row, col] * target @ c_inverse @ target
                alpha[row, col] = max(up / down, 0)
            else:
                alpha[row, col] = np.nan

# 指定输出的TIFF文件名
output_tiff_file = "C:\\Users\\RS\\\Desktop\\EMIT\\Result\\EMIT_L1B_RAD_001_20230420T060148_2311004_021_subset_iter20.tif"
# 获取数组的维度
rows, cols = alpha.shape

# 创建一个新的TIFF文件
driver = gdal.GetDriverByName('GTiff')
dataset = driver.Create(output_tiff_file, cols, rows, 1, gdal.GDT_Float32)

# 将NumPy数组写入TIFF文件
band = dataset.GetRasterBand(1)
band.WriteArray(alpha)

# 设置获取的地理参考信息
dataset.SetGeoTransform(geo_transform)
dataset.SetProjection(projection)
