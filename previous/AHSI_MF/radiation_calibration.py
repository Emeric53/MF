'''
    基于该代码实现对AHSI数据的辐射定标工作。
'''
from osgeo import gdal
import numpy as np

# 设置SWIR 短波红外文件路径 
file_path = r"H:\AHSI_part2\GF5B_AHSI_E74.4_N37.2_20231009_011103_L10000403537\GF5B_AHSI_E74.4_N37.2_20231009_011103_L10000403537_SW.tif"
# 利用gdal打开数据
dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
# 从已存在的TIFF文件中获取地理参考信息
geo_transform = dataset.GetGeoTransform()
projection = dataset.GetProjection()

# 生成存储辐射定标参数的列表 
factorlist=[]
with open('AHSI_MF\GF5B_AHSI_RadCal_SWIR.raw', 'r') as radiation_calibration_file:
    result = radiation_calibration_file.readlines()
    for i in result:
        factor = i.split(',')[0]
        residual = i.split(',')[1].rstrip('\n')
        factorlist.append([factor, residual])
print(factorlist)

# 获取所有波段数目
num_bands = dataset.RasterCount
print(num_bands)

# 定义数组存储各波段数据
band_data_list = []

# 遍历各个波段，注意数据集的getrasterband方法索引从1开始
for band_index in range(1, 20):
    # 依据索引获取当前波段数据
    current_band = dataset.GetRasterBand(band_index)
    # 读取波段数据为NumPy数组
    current_band_data =np.array(current_band.ReadAsArray(),dtype=np.float32)
    mask = np.where(current_band_data>0)
    current_band_data[mask] = current_band_data[mask]*float(factorlist[band_index-1][0]) + float(factorlist[band_index-1][1])
    # 添加入波段数据总和中
    band_data_list.append(current_band_data)
