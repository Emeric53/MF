import numpy as np
from osgeo import gdal

# 打开栅格影像
dataset = gdal.Open("C:\\Users\\RS\\Desktop\\MethaneColumnConcentration\\EMIT_L1B_RAD_001_20230806T035031_2321803_033_50.tiff")

# 读取栅格数据
raster = dataset.ReadAsArray()

image_data = np.array(raster)
array_without_nan = np.nan_to_num(image_data, nan=-np.inf)
# 使用argsort函数获取排序后的索引
sorted_indices = np.argsort(array_without_nan, axis=None)
# 获取前10个最大值的索引
top10_indices = sorted_indices[-100:]
# 将一维索引转换为二维索引
row_indices, col_indices = np.unravel_index(top10_indices, array_without_nan.shape)

file_path = "C:\\Users\\RS\\Documents\\DataProcessing\\data\\EMIT\\EMIT_L1B_RAD_001_20230806T035031_2321803_033.tif"
dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
# 从已存在的TIFF文件中获取地理参考信息
geo_transform = dataset.GetGeoTransform()
projection = dataset.GetProjection()

# 获取波段数目
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
# 将波段数据转为array
image_data = np.array(band_data_list)
# 以每个波段为分割，计算非nan的平均值
u = np.nanmean(image_data, axis=(1, 2))
# 获取栅格数据的波段，行，列
bands, rows, cols = image_data.shape
vectors = []
for i in range(len(row_indices)):
    row = row_indices[i]
    col = col_indices[i]
    a = array_without_nan[row, col]
    l = image_data[:, row, col]
    result = (l-u)/(u*a)
    vectors.append(result)

average_vector = np.mean(vectors, axis=0)
# 打开文件并写入列表内容
with open('output.txt', 'w') as file:
    for item in average_vector:
        file.write(str(item) + '\n')
