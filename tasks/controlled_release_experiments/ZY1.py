from osgeo import gdal
import numpy as np
from matplotlib import pyplot as plt

import time
import math
import tempfile

from methane_retrieval_algorithms.matchedfilter import matched_filter as mf
from methane_retrieval_algorithms.columnwise_matchedfilter import (
    columnwise_matched_filter as cmf,
)
from methane_retrieval_algorithms.ml_matchedfilter import ml_matched_filter as mlmf
from methane_retrieval_algorithms.columnwise_ml_matchedfilter import (
    columnwise_ml_matched_filter as cmlmf,
)
from utils import satellites_data as sd
from utils import generate_radiance_lut_and_uas as glut


def ZY1_test(filepath):
    _, radiance_cube = sd.ZY1_data.get_ZY1_radiances_from_dat(filepath, 2150, 2500)
    sza, altitude = sd.ZY1_data.get_sza_altitude(filepath)
    if altitude > 5:
        altitude = 5
    _, uas = glut.generate_satellite_uas_for_specific_range_from_lut(
        "ZY1", 0, 50000, 2150, 2500, sza, altitude
    )

    # 原始匹配滤波算法结果测试
    mf_enhancement = mf(radiance_cube, uas, True, True, True)
    cmf_enhancement = cmf(radiance_cube, uas, True, True, True, 5)

    # 多层匹配滤波算法结果测试
    # methane_enhancement_mlmf = mfs.ml_matched_filter_new(radiance_cube,uas, True)
    # np.save("methane_enhancement.npy",methane_enhancement)

    # 输出结果到tiff文件
    mf_outputfilepath = filepath.replace(".dat", "_mf.tif")
    cmf_outputfilepath = filepath.replace(".dat", "_cmf.tif")
    export_result_to_tiff(filepath, mf_enhancement, mf_outputfilepath)
    export_result_to_tiff(filepath, cmf_enhancement, cmf_outputfilepath)

    return mf_enhancement, cmf_enhancement


def export_result_to_tiff(filepath, result_array, outputfilepath):
    # 打开原始影像文件
    original_dataset = gdal.Open(filepath)

    rpc_info = original_dataset.GetMetadata("RPC")

    # 创建一个临时文件来存储结果数组
    temp_file = tempfile.NamedTemporaryFile(suffix=".tif").name
    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(
        temp_file, result_array.shape[1], result_array.shape[0], 1, gdal.GDT_Float32
    )

    # 将结果数组写入临时影像
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(result_array)
    output_band.FlushCache()

    # 设置临时影像的地理信息
    output_dataset.SetProjection(original_dataset.GetProjection())
    output_dataset.SetGeoTransform(original_dataset.GetGeoTransform())

    # 将RPC信息写入临时文件的元数据
    output_dataset.SetMetadata(rpc_info, "RPC")

    # 关闭并保存临时影像文件
    output_dataset = None

    # 使用 WarpOptions 并从临时文件路径进行校正
    corrected_file = outputfilepath
    warp_options = gdal.WarpOptions(rpc=True)
    gdal.Warp(corrected_file, temp_file, options=warp_options)


# ZY1 文件路径
filepath = [
    "/home/emeric/Documents/stanford/ZY1/ZY1F_AHSI_W111.72_N33.06_20221026_004370_L1A0000265656_VNSW_Rad.dat",
    "/home/emeric/Documents/stanford/ZY1/ZY1F_AHSI_W111.74_N33.06_20221023_004327_L1A0000261518_VNSW_Rad.dat",
    "/home/emeric/Documents/stanford/ZY1/ZY1F_AHSI_W111.85_N32.62_20221020_004284_L1A0000258234_VNSW_Rad.dat",
]

for file in filepath:
    start_time = time.time()
    ZY1_test(file)
    end_time = time.time()
    print("Time cost: ", end_time - start_time)
