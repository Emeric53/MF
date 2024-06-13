import numpy as np
from osgeo import gdal
import math
from scipy.integrate import quad
import math
import pathlib
""" This code is used to calculate the cross section flux of the plume.
by computing the flux through one or more plume cross sections orthogonal to the plume axis.
"""

# sourcerate = the intergral of the product of the plume concentration and the wind speed over the plume cross section
# Set the path of the plume
plume_filepath = r"C:\Users\RS\Desktop\EMIT\MethanePlume\EMIT_L2B_CH4PLM_001_20230204T041009_000618_tiff.tif"

# Read the plume data and remove the invalid values
plume_data = gdal.Open(plume_filepath, gdal.GA_ReadOnly)
plume_data = plume_data.ReadAsArray()

# Set the pixel resolution, unit: m
pixel_resolution = 30

# 当前像元的行列号
original_row = 100
original_col = 100

# 指定风向的角度，例如45度
wind_direction_degrees = 45

# 移动步长
step_size = 1  # 假设每次移动一个像元的距离

#基于风向和步长计算目标像元的行列号
def move(current_row, current_col, wind_direction_degrees, step_size):
    if wind_direction_degrees == 0:
        delta_row = 0*step_size
        delta_col = 1*step_size
    elif wind_direction_degrees == 45:
        delta_row = -1*step_size
        delta_col = 1*step_size
    elif wind_direction_degrees == 90:
        delta_row = -1*step_size
        delta_col = 0*step_size
    elif wind_direction_degrees == 135:
        delta_row = -1*step_size
        delta_col = -1*step_size
    elif wind_direction_degrees == 180:
        delta_row = 0*step_size
        delta_col = -1*step_size
    elif wind_direction_degrees == 225:
        delta_row = 1*step_size
        delta_col = -1*step_size
    elif wind_direction_degrees == 270:
        delta_row = 1*step_size
        delta_col = 0*step_size
    elif wind_direction_degrees == 315:
        delta_row = 1*step_size
        delta_col = 1*step_size
    else:
        # 将角度转换为弧度
        wind_direction_radians = math.radians(wind_direction_degrees)

        if abs(math.sin(wind_direction_radians)) > abs(math.cos(wind_direction_radians)):
            # 计算相邻像元的行列号
            delta_row = - step_size*math.copysign(1, math.sin(wind_direction_radians))
            delta_col = int(round(step_size*(1/math.tan(wind_direction_radians))))
        else:
            # 计算相邻像元的行列号
            delta_col = step_size * math.copysign(1, math.cos(wind_direction_radians))
            delta_row = - int(round(step_size * math.tan(wind_direction_radians)))
    new_row = current_row + delta_row
    new_col = current_col + delta_col
    return new_row, new_col

#预定义当前像元的行列号
current_row = original_row
current_col = original_col

#设置遍历的条件
while 0 < current_row < 200 and 0 < current_col < 200:
    current_row, current_col = move(original_row, original_col, wind_direction_degrees, step_size)
    print(current_row, current_col)
    step_size += 1
