import os
import sys

sys.path.append('C:\\Users\\RS\\Documents\\DataProcessing\\python\\modules\\emit_tools.py')
import emit_tools as et
from osgeo import gdal

filepath = ("C:\\Users\\RS\\Desktop\\EMIT\\Imageswithplume\\nc\\EMIT_L1B_RAD_001_20230204T041009_2303503_016.nc")
outpath = "C:\\Users\\RS\\Desktop\\EMIT\\Imageswithplume\\envi"
if not os.path.exists(outpath):
    os.makedirs(outpath)
ds = et.emit_xarray(filepath, ortho=True)
et.write_envi(ds, outpath, overwrite=False, extension='.img', interleave='BIL', glt_file=False)

# 输入文件名和输出文件名
input_envi = "C:\\Users\\RS\\Desktop\\EMIT\\Imageswithplume\\envi\\EMIT_L1B_RAD_001_20230204T041009_2303503_016_radiance.img"
output_tif = "C:\\Users\\RS\\Desktop\\EMIT\\Imageswithplume\\tiff\\EMIT_L1B_RAD_001_20230204T041009_2303503_016_radiance.tiff"

# 打开 ENVI 格式文件
input_ds = gdal.Open(input_envi)
print(input_ds)

# 将 ENVI 格式文件转换为 TIFF 格式
gdal.Translate(output_tif, input_ds, format='GTiff')

# 关闭文件
input_ds = None
