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


def GF5B_retrieval(filepath):
    # 读取GF5B数据
    _, radiance_cube = sd.GF5B_data.get_AHSI_radiances_from_dat(filepath, 2150, 2500)
    # 获取GF5B数据的SZA和高度
    sza, altitude = sd.GF5B_data.get_sza_altitude_from_dat(filepath)
    if altitude > 5:
        altitude = 5
    # 生成GF5B的单位吸收谱
    _, uas = glut.generate_satellite_uas_for_specific_range_from_lut(
        "AHSI", 0, 50000, 2150, 2500, sza, altitude
    )

    # 原始匹配滤波算法结果测试
    mf_enhancement = mf(radiance_cube, uas, True, True, True)
    cmf_enhancement = cmf(radiance_cube, uas, True, True, True, 5)

    # 多层匹配滤波算法结果测试
    # mlmf_enhancement = mlmf(radiance_cube,uas,"AHSI",True,True,True)

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


filepath = [
    "/home/emeric/Documents/stanford/GF5B/GF5B_AHSI_W112.1_N32.8_20221115_006332_L10000239663_VNSW_Rad.dat"
]


# 获取反演结果
for file in filepath:
    start_time = time.time()
    mf_methane_enhancement, cmf_methane_enhancement = GF5B_retrieval(file)
    end_time = time.time()
    print("Time cost: ", end_time - start_time)


# 量化排放速率

# 指定提取甲烷烟羽的tiff文件路径
plume_path = ""
# 使用IME模型估算排放速率
