import pathlib as pl
import os
import sys


sys.path.append("C://Users//RS//VSCode//matchedfiltermethod//src")
from algorithms import columnwise_matchedfilter

from utils import satellites_data as sd
from utils import generate_radiance_lut_and_uas as glut


# 单个GF5B文件处理
def single_EMIT_run(filepath, outputfolder):
    low_wavelength = 2150
    high_wavelength = 2500
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return
    emit_bands, EMIT_radiance = sd.EMIT_data.get_emit_bands_array(
        filepath, low_wavelength, high_wavelength
    )

    # 读取 sza，地表高程的参数
    sza, altitude = sd.EMIT_data.get_sza_altitude(filepath)
    # 生成初始单位吸收谱 用于计算
    _, uas = glut.generate_satellite_uas_for_specific_range_from_lut(
        "EMIT", 0, 50000, low_wavelength, high_wavelength, sza, altitude
    )
    print(sza, altitude)
    try:
        enhancement = columnwise_matchedfilter.columnwise_matched_filter(
            EMIT_radiance, uas, True, True
        )
        sd.EMIT_data.export_emit_array_to_nc_tif(enhancement, filepath, outputfolder)
    except Exception as e:
        print("Error in processing: ", filepath)
        print(e)


def batch_EMIT_run(outputfolder, province_region):
    # 结果文件夹
    # outputfolder = "J:\\shanxi_result"
    filefolder = "J:\EMIT\L1B"
    # 用于存储符合条件的文件路径
    filepathlist = []

    # 遍历文件夹
    for root, dirs, files in os.walk(filefolder):
        for file in files:
            # 检查文件是否是 .nc 后缀并且文件名中包含 _RAD_
            if file.endswith(".nc") and "_RAD_" in file:
                # 获取文件的完整路径
                file_path = os.path.join(root, file)
                # 将符合条件的文件路径添加到数组中
                filepathlist.append(file_path)
    # 遍历文件路径
    for filepath in filepathlist:
        if os.path.exists(filepath) is False:
            continue
        if sd.EMIT_data.is_within_region(filepath, province_region) is True:
            print(f"current file: {filepath} is in targeted province")
            single_EMIT_run(filepath, outputfolder)


# test single file
filepath = r"C:\\Users\\RS\\Desktop\\Lifei_essay_data\\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985\\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985_SW.tif"
outputfolder = r"C:\\Users\RS\\Desktop\\hi"
if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
# single_GF5B_run(filepath, outputfolder)


shanxi_region = [111.0, 114.5, 34.5, 40.8]
outputfolder = r"L:\shanxi_emit"
if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
batch_EMIT_run(outputfolder, shanxi_region)

xinjiang_region = [80.0, 95.0, 35.0, 49.0]
outputfolder = r"L:\xinjiang_emit"
if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
batch_EMIT_run(outputfolder, xinjiang_region)

neimenggu_region = [97.0, 126.0, 37.0, 53.0]
outputfolder = r"L:\neimenggu_emit"
if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
batch_EMIT_run(outputfolder, neimenggu_region)
