import os
import sys
sys.path.append("C:\\Users\\RS\\Documents\\DataProcessing\\python\\modules\\")

from osgeo import gdal
import pathlib
import emit_tools as et

#设置文件路径和输出路径
filefolder = "F:\\EMIT\\nc"
outpath = "F:\\EMIT\\envi"
outtifpath = "F:\\EMIT\\tiff"
filelist = pathlib.Path(filefolder).glob('*.nc')

for filepath in filelist:
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    #读取dataset并写为envi文件
    ds = et.emit_xarray(filepath, ortho=True)
    et.write_envi(ds, outpath, overwrite=False, extension='.img', interleave='BIL', glt_file=False)

    # 设置envi文件路径和tiff输出路径
    input_envi = pathlib.Path(outpath).parent / (pathlib.Path(filepath).stem + '_radiance.img')
    output_tif = pathlib.Path(outtifpath).parent / (pathlib.Path(filepath).stem + '_radiance.tif')

    # 打开 ENVI 格式文件
    input_ds = gdal.Open(input_envi)
    print(input_ds)

    # 将 ENVI 格式文件转换为 TIFF 格式
    gdal.Translate(output_tif, input_ds, format='GTiff')

    # 关闭文件
    input_ds = None
