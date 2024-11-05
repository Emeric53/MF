import pathlib as pl
import os
import sys

sys.path.append("C://Users//RS//VSCode//matchedfiltermethod//src")
from algorithms import matched_filter_variants as mfs
from utils import satellites_data as sd
from utils import generate_radiance_lut_and_uas as glut


# mf_type 以数字代表使用的匹配滤波算法 类型
# 0：columnwise + 迭代 + 反射率校正因子 的匹配滤波算法


# 部分参数设置
low_wavelength = 2150
high_wavelength = 2500


# 单个GF5B文件处理
def run_for_GF5B(filepath, outputfolder, mf_type):
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
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
        process_radiance_by_mf(AHSI_radiance, uas, mf_type)
    except Exception as e:
        print("Error in processing: ", filepath)
        print(e)


# 单个AHSI文件处理
def run_for_ZY1(filepath, outputfolder, mf_type):
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return
    # 基于 波段范围 读取辐射定标后的radiance的cube
    _, AHSI_radiance = sd.GF5B_data.get_calibrated_radiance(
        filepath, low_wavelength, high_wavelength
    )
    # 读取 sza，地表高程的参数
    sza, altitude = sd.GF5B_data.get_sza_altitude(filepath)
    # 生成初始单位吸收谱 用于计算
    uas = glut.generate_satellite_uas_for_specific_range_from_lut(
        "AHSI", 0, 50000, low_wavelength, high_wavelength, sza, altitude
    )
    process_radiance_by_mf(AHSI_radiance, uas, mf_type)


# 单个EMIT文件处理
def run_for_EMIT(filepath, outputfolder, mf_type):
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return
    emit_bands, EMIT_radiance = sd.EMIT_data.get_emit_bands_array(filepath, 2100, 2500)
    uas_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\EMIT_unit_absorption_spectrum.txt"
    interval_uas_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_unit_absorption_spectrum.txt"
    process_radiance_by_mf(EMIT_radiance, uas_path, interval_uas_path, mf_type)


# 单个EnMAP文件处理
def run_for_EnMAP(filepath, outputfolder, mf_type):
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return
    EnMAP_bands, EnMAP_radiance = sd.EnMAP_data.get_enmap_bands_array(
        filepath, 2100, 2500
    )
    uas_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\EnMAP_unit_absorption_spectrum.txt"
    process_radiance_by_mf(EnMAP_radiance, uas_path, mf_type)


# 单个PRISMA文件处理
def run_for_PRISMA(filepath, outputfolder, mf_type):
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return
    EnMAP_bands, radiance = sd.PRISMA_data.get_prisma_bands_array(filepath, 2100, 2500)
    uas_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\PRISMA_unit_absorption_spectrum.txt"

    enhancement_result = process_radiance_by_mf(
        radiance, uas_path, filepath, outputfolder, mf_type
    )
    return enhancement_result


# 获取文件夹中的所有子文件夹
def get_subdirectories(folder_path: str):
    """
    获取指定文件夹中所有子文件夹的路径列表。
    :param  folder_path: 父文件夹的路径。
    :return: 子文件夹路径列表, 子文件夹名称列表
    """
    dir_paths = [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name))
    ]
    dir_names = [
        name
        for name in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, name))
    ]
    return dir_paths, dir_names


# 批量运行
def runinbatch(satellite_name: str):
    if satellite_name == "AHSI":
        outputfolder = "J:\\AHSI_result"
        filefolder_list = [
            "I:\\AHSI_part2",
            "K:\\AHSI_part1",
            "M:\\AHSI_part3",
            "J:\\AHSI_part4",
        ]
        process_files(filefolder_list, outputfolder, "AHSI")

    elif satellite_name == "EMIT":
        radiance_folder = "I:\\EMIT\\rad"
        result_folder = "I:\\EMIT\\methane_result\\Direct_result"
        process_files([radiance_folder], result_folder, "EMIT")
    else:
        print("Invalid satellite name, please select from 'AHSI' or 'EMIT'.")


# public 处理函数
def process_files(filefolder_list, outputfolder, satellite_name):
    if satellite_name == "AHSI":
        for filefolder in filefolder_list:
            filelist, namelist = get_subdirectories(filefolder)
            for index in range(len(filelist)):
                filepath = os.path.join(filelist[index], namelist[index] + "_SW.tif")
                if os.path.exists(filepath) is False:
                    continue
                print("Current file: ", filepath)
                run_for_GF5B(filepath, outputfolder, mf_type=0)

    elif satellite_name == "EMIT":
        radiance_path_list = pl.Path(filefolder_list[0]).glob("*.nc")
        outputfile_list = [str(f.name) for f in pl.Path(outputfolder).glob("*.nc")]
        for radiance_path in radiance_path_list:
            current_filename = radiance_path.name
            if current_filename in outputfile_list:
                continue
            run_for_EMIT(radiance_path, outputfolder, mf_type=0)


# 通用的radiance处理函数
def process_radiance_by_mf(radiance_cube, uas, mf_type):
    if mf_type == 0:
        enhancement = mfs.columnwise_matched_filter(radiance_cube, uas, True, True)
    elif mf_type == 1:
        enhancement = mfs.ml_matched_filter(radiance_cube, uas, True)
    else:
        print("Invalid mf_type: 0 for original mf and 1 for modified mf")
        return
    return enhancement


# 各种卫星数据批量运算


def prisma():
    prisma_folder = "I:\\PRISMA"
    outputfolder = "I:\\PRISMA_result"
    if os.path.exists(outputfolder) is False:
        os.mkdir(outputfolder)
    filelist, namelist = get_subdirectories(prisma_folder)
    for index in range(len(filelist)):
        filepath = os.path.join(filelist[index], namelist[index] + ".tif")
        run_for_PRISMA(filepath, outputfolder, mf_type=0)


def gf5b():
    gf5b_folder = "I:\\GF5B"
    outputfolder = "I:\\GF5B_result"
    if os.path.exists(outputfolder) is False:
        os.mkdir(outputfolder)
    filelist, namelist = get_subdirectories(gf5b_folder)
    for index in range(len(filelist)):
        filepath = os.path.join(filelist[index], namelist[index] + ".tif")
        run_for_GF5B(filepath, outputfolder, mf_type=0)


def zy1():
    gf5b_folder = "I:\\GF5B"
    outputfolder = "I:\\GF5B_result"
    if os.path.exists(outputfolder) is False:
        os.mkdir(outputfolder)
    filelist, namelist = get_subdirectories(gf5b_folder)
    for index in range(len(filelist)):
        filepath = os.path.join(filelist[index], namelist[index] + ".tif")
        run_for_ZY1(filepath, outputfolder, mf_type=0)


def enmap():
    enmap_folder = "I:\\EnMAP"
    outputfolder = "I:\\EnMAP_result"
    if os.path.exists(outputfolder) is False:
        os.mkdir(outputfolder)
    filelist, namelist = get_subdirectories(enmap_folder)
    for index in range(len(filelist)):
        filepath = os.path.join(filelist[index], namelist[index] + ".tif")
        run_for_EnMAP(filepath, outputfolder, mf_type=0)


def emit():
    enmap_folder = "I:\\EnMAP"
    outputfolder = "I:\\EnMAP_result"
    if os.path.exists(outputfolder) is False:
        os.mkdir(outputfolder)
    filelist, namelist = get_subdirectories(enmap_folder)
    for index in range(len(filelist)):
        filepath = os.path.join(filelist[index], namelist[index] + ".tif")
        run_for_EnMAP(filepath, outputfolder, mf_type=0)


if __name__ == "__main__":
    # runinbatch("AHSI")
    print("All done!")
    filepath = "I:\AHSI_part2\GF5B_AHSI_E78.1_N36.7_20231004_011030_L10000400463\GF5B_AHSI_E78.1_N36.7_20231004_011030_L10000400463_SW.tif"
    radiance_cube = sd.GF5B_data.get_ahsi_array(filepath)
    print(radiance_cube.shape)
