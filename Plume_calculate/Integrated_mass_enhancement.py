import numpy as np
from osgeo import gdal
import math

# set the filepath of methane plume image or the enhancement of methane
plume_filepath = r"C:\Users\RS\Desktop\EMIT_L2B_CH4PLM_001_20230420T060148_000837.tif"

# read the array of the plume
plume_data = gdal.Open(plume_filepath, gdal.GA_ReadOnly)
plume_data = plume_data.ReadAsArray()

# set the resolution of the pixel with the unit of meter
pixel_resolution = 60
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
IME = (np.sum(values)*pixel_area)
print(IME/8000)
# 如何进行量纲的转换   ppm·m to ppm to kg/m2
IME = IME*0.01604*3600*8000/22.4/1000000

# get the windspeed at 10m m/s
windspeed_10 = 3.5

# set the parameters of the formula
a = 0.37
b = 0.64

# calculate the efficitive windspeed with the formula
efficitive_windspeed = a*windspeed_10 + b

# calculate the emission rate of the plume  kg/h
q = efficitive_windspeed*IME/plume_L

print(plume_filepath+" Emission rate: "+str(q))

