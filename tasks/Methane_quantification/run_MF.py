import pathlib as pl
import os
import sys

sys.path.append("C://Users//RS//VSCode//matchedfiltermethod//src")
from algorithms import matched_filter_variants as mf
from utils import satellites_data as sd


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
        outputfolder = "I:\\AHSI_result"
        filefolder_list = [
            "F:\\AHSI_part1",
            "H:\\AHSI_part2",
            "L:\\AHSI_part3",
            "I:\\AHSI_part4",
        ]
        process_files(filefolder_list, outputfolder, "AHSI")

    elif satellite_name == "EMIT":
        radiance_folder = "I:\\EMIT\\rad"
        result_folder = "I:\\EMIT\\methane_result\\Direct_result"
        process_files([radiance_folder], result_folder, "EMIT")
    else:
        print("Invalid satellite name, please select from 'AHSI' or 'EMIT'.")


# 公用处理函数
def process_files(filefolder_list, outputfolder, satellite_name):
    if satellite_name == "AHSI":
        for filefolder in filefolder_list:
            filelist, namelist = get_subdirectories(filefolder)
            for index in range(len(filelist)):
                filepath = os.path.join(filelist[index], namelist[index] + "_SW.tif")
                rumfor_AHSI(filepath, outputfolder, mf_type=0)

    elif satellite_name == "EMIT":
        radiance_path_list = pl.Path(filefolder_list[0]).glob("*.nc")
        outputfile_list = [str(f.name) for f in pl.Path(outputfolder).glob("*.nc")]
        for radiance_path in radiance_path_list:
            current_filename = radiance_path.name
            if current_filename in outputfile_list:
                continue
            rumfor_EMIT(radiance_path, outputfolder, mf_type=0)


# 单个AHSI文件处理
def runfor_AHSI(filepath, outputfolder, mf_type):
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return
    _, AHSI_radiance = sd.AHSI_data.get_calibrated_radiance(filepath, 2100, 2500)
    uas_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum.txt"

    enhancement_result = process_radiance_by_mf(AHSI_radiance, uas_path, mf_type)


# 单个EMIT文件处理
def runfor_EMIT(filepath, outputfolder, mf_type):
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return
    emit_bands, EMIT_radiance = sd.EMIT_data.get_emit_bands_array(filepath, 2100, 2500)
    uas_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\EMIT_unit_absorption_spectrum.txt"
    interval_uas_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_unit_absorption_spectrum.txt"
    process_radiance_by_mf(EMIT_radiance, uas_path, interval_uas_path, mf_type)


# 单个EnMAP文件处理
def runfor_EnMAP(filepath, outputfolder, mf_type):
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
def runfor_PRISMA(filepath, outputfolder, mf_type):
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return
    EnMAP_bands, radiance = sd.PRISMA_data.get_prisma_bands_array(filepath, 2100, 2500)
    uas_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\PRISMA_unit_absorption_spectrum.txt"

    enhancement_result = process_radiance_by_mf(
        radiance, uas_path, filepath, outputfolder, mf_type
    )


# 通用的radiance处理函数
def process_radiance_by_mf(radiance_cube, uas_path, mf_type):
    bands, uas = sd.general_functions.open_unit_absorption_spectrum(
        uas_path, 2100, 2500
    )
    if mf_type == 0:
        enhancement = mf.columnwise_matched_filter(radiance_cube, uas, True, True)
    elif mf_type == 1:
        enhancement = mf.ml_matched_filter(radiance_cube, uas, True)
    else:
        print("Invalid mf_type: 0 for original mf and 1 for modified mf")
        return
    return enhancement


def prisma():
    prisma_folder = "I:\\PRISMA"

    outputfolder = "I:\\PRISMA_result"
    if os.path.exists(outputfolder) is False:
        os.mkdir(outputfolder)
    filelist, namelist = get_subdirectories(prisma_folder)
    for index in range(len(filelist)):
        filepath = os.path.join(filelist[index], namelist[index] + ".tif")
        runfor_PRISMA(filepath, outputfolder, mf_type=0)


def gf5b():
    gf5b_folder = "I:\\GF5B"
    outputfolder = "I:\\GF5B_result"
    if os.path.exists(outputfolder) is False:
        os.mkdir(outputfolder)
    filelist, namelist = get_subdirectories(gf5b_folder)
    for index in range(len(filelist)):
        filepath = os.path.join(filelist[index], namelist[index] + ".tif")
        runfor_AHSI(filepath, outputfolder, mf_type=0)


def zy1():
    gf5b_folder = "I:\\GF5B"
    outputfolder = "I:\\GF5B_result"
    if os.path.exists(outputfolder) is False:
        os.mkdir(outputfolder)
    filelist, namelist = get_subdirectories(gf5b_folder)
    for index in range(len(filelist)):
        filepath = os.path.join(filelist[index], namelist[index] + ".tif")
        runfor_AHSI(filepath, outputfolder, mf_type=0)


def enmap():
    enmap_folder = "I:\\EnMAP"
    outputfolder = "I:\\EnMAP_result"
    if os.path.exists(outputfolder) is False:
        os.mkdir(outputfolder)
    filelist, namelist = get_subdirectories(enmap_folder)
    for index in range(len(filelist)):
        filepath = os.path.join(filelist[index], namelist[index] + ".tif")
        runfor_EnMAP(filepath, outputfolder, mf_type=0)


if __name__ == "__main__":
    prisma()
    gf5b()
    zy1()
    enmap()
