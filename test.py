"""该代码用于进行快速的 语法测试 和 函数与方法测试 """
from osgeo import gdal
import numpy as np
import xarray as xr

filepath = "C:\\Users\\RS\\Desktop\\EMIT_L1B_RAD_001_20240218T044027_2404903_012.nc"
dataset = xr.open_dataset(filepath)
radiance = dataset['radiance'][-1,:,:]
loc = xr.open_dataset(filepath,group='location')
# 获取 lat 和 lon 数据
lat = loc['lat']
lon = loc['lon']

print(radiance.shape, lat.shape, lon.shape)

radiance_geo = xr.DataArray(
    radiance.values,
    dims=("downtrack", "crosstrack"),
    coords={"lat": lat, "lon": lon},
)

radiance_geo.to_raster("C:\\Users\\RS\\Desktop\\EMIT_L1B_RAD_001_20240218T044027_2404903_012_test.tif")


