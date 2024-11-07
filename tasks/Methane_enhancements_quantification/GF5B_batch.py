import pathlib as pl
import os
import sys
import re
import shutil


sys.path.append("C://Users//RS//VSCode//matchedfiltermethod//src")
from algorithms import columnwise_matchedfilter

from utils import satellites_data as sd
from utils import generate_radiance_lut_and_uas as glut


# 单个GF5B文件处理
def single_GF5B_run(filepath, outputfolder):
    low_wavelength = 2150
    high_wavelength = 2500
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        # os.remove(outputfile)
        return
    # 基于 波段范围 读取辐射定标后的radiance的cube
    _, AHSI_radiance = sd.GF5B_data.get_calibrated_radiance(
        filepath, low_wavelength, high_wavelength
    )
    # 读取 sza，地表高程的参数
    sza, altitude = sd.GF5B_data.get_sza_altitude(filepath)
    # 生成初始单位吸收谱 用于计算
    _, uas = glut.generate_satellite_uas_for_specific_range_from_lut(
        "AHSI", 0, 50000, low_wavelength, high_wavelength, sza, altitude
    )
    try:
        enhancement = columnwise_matchedfilter.columnwise_matched_filter(
            AHSI_radiance, uas, True, True
        )
        output_rpb = outputfile.replace(".tif", ".rpb")
        if not os.path.exists(output_rpb):
            original_rpb = filepath.replace(".tif", ".rpb")
            shutil.copy(original_rpb, output_rpb)
        sd.GF5B_data.export_ahsi_array_to_tiff(
            enhancement, filepath, outputfolder, output_filename=None, orthorectify=True
        )
    except Exception as e:
        print("Error in processing: ", filepath)
        print(e)


def is_within_province(data_name, province_region):
    lon_min, lon_max, lat_min, lat_max = province_region[:]
    match = re.search(r"E([+-]?\d+\.\d+).*N([+-]?\d+\.\d+)", data_name)
    if match:
        longitude = float(match.group(1))
        latitude = float(match.group(2))
        # 判断经纬度是否在指定省份范围内
        if lon_min <= longitude <= lon_max and lat_min <= latitude <= lat_max:
            return True
    return False


def get_swir_filepath(folder_path: str):
    """
    获取指定文件夹中所有子文件夹的路径列表。
    :param  folder_path: 父文件夹的路径。
    :return: 子文件夹路径列表, 子文件夹名称列表
    """
    dir_paths = [
        os.path.join(folder_path, name, name + "_SW.tif")
        for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name))
    ]
    return dir_paths


def batch_GF5B_run(outputfolder, province_region):
    # 结果文件夹
    # outputfolder = "J:\\shanxi_result"
    filefolder_list = [
        "I:\\AHSI_part2",
        "K:\\AHSI_part1",
        "M:\\AHSI_part3",
        "J:\\AHSI_part4",
    ]
    for filefolder in filefolder_list:
        print("Current folder: ", filefolder)
        filepathlist = get_swir_filepath(filefolder)
        for filepath in filepathlist:
            if os.path.exists(filepath) is False:
                continue
            if is_within_province(filepath, province_region) is True:
                print(f"current file {filepath} is in targeted province")
                single_GF5B_run(filepath, outputfolder)


filepath = r"C:\Users\RS\Desktop\Lifei_essay_data\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985_SW.tif"
outputfolder = r"C:\Users\RS\Desktop\hi"

if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
# single_GF5B_run(filepath, outputfolder)


shanxi_region = [111.0, 114.5, 34.5, 40.8]
outputfolder = r"L:\shanxi_gf5b"
if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
batch_GF5B_run(outputfolder, shanxi_region)

xinjiang_region = [80.0, 95.0, 35.0, 49.0]
outputfolder = r"L:\xinjiang_gf5b"
if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
batch_GF5B_run(outputfolder, xinjiang_region)

neimenggu_region = [97.0, 126.0, 37.0, 53.0]
outputfolder = r"L:\neimenggu_gf5b"
if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
batch_GF5B_run(outputfolder, neimenggu_region)
