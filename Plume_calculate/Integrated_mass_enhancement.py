import numpy as np
from osgeo import gdal
import math

# set the filepath of methane plume image or the enhancement of methane
#plume_filepath = r"J:\GF5B_AHSI_W104.1_N32.8_20220209_002267_L10000074984\result\bigplume.tif"
plume_filepath = r"J:\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985\result\bigplume.tif"
# read the array of the plume
plume_data = gdal.Open(plume_filepath, gdal.GA_ReadOnly)
plume_data = plume_data.ReadAsArray()

print(np.max(plume_data))
# set the resolution of the pixel with the unit of meter
pixel_resolution = 30
pixel_area = pixel_resolution*pixel_resolution

# cal the area and the length of the plume
nan_count = np.count_nonzero(~np.isnan(plume_data))
plume_area = nan_count*pixel_area
plume_L = math.sqrt(plume_area)

# get all the values of the plume
cols, rows = plume_data.shape
values = []
for col in range(cols):
    for row in range(rows):
        if plume_data[col][row] != -9999:
            values.append(plume_data[col][row])

# calculate the integrated mass enhancement
IME = np.sum(values)
print(IME)
# 如何进行量纲的转换   ppm to kg/m2
IME = IME*5.155*3600

# get the windspeed at 10m m/s
windspeed_10 = 1.57

# set the parameters of the formula
a = 0.38
b = 0.41

# calculate the efficitive windspeed with the formula
efficitive_windspeed = a*windspeed_10 + b

print(plume_L)
# calculate the emission rate of the plume  kg/h
q = efficitive_windspeed*IME/plume_L

print(plume_filepath+" Emission rate: "+str(q))

