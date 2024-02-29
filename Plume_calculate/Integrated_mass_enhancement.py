import numpy as np
from osgeo import gdal
import math

# set the filepaht of methane plume image
plume_filepath = r"C:\Users\RS\Desktop\EMIT\MethanePlume\EMIT_L2B_CH4PLM_001_20230204T041009_000618_tiff.tif"
plume_filepath = plume_filepath.replace('\\', '/')

# read the array of the plume
plume_data = gdal.Open(plume_filepath, gdal.GA_ReadOnly)
plume_data = plume_data.ReadAsArray()

#设置 像元的分辨率大小，单位为 m
pixel_resolution = 30
pixel_area = pixel_resolution*pixel_resolution

#统计烟羽的面积和尺度参数
nan_count = np.count_nonzero(~np.isnan(plume_data))
plume_area = nan_count*pixel_area
plume_L = math.sqrt(plume_area)

#将烟羽的数据转换为一维数组，然后剔除掉无效值
cols, rows = plume_data.shape
values = []

for col in range(cols):
    for row in range(rows):
        if plume_data[col][row] != -9999:
            values.append(plume_data[col][row])

#根据IME的定义 计算IME
IME = (np.sum(values)*pixel_area)/plume_area
print(IME)

#基于有效风速和尺度参数以及IME计算烟羽的排放量
windspeed = 1
a = 0.37
b = 0.64
efficitive_windspeed = a*windspeed + b

q = efficitive_windspeed*IME/plume_L
print(q)


