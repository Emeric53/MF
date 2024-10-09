import pathlib as pl
import os
import sys

sys.path.append("C://Users//RS//VSCode//matchedfiltermethod//src")
from algorithms import matchedfilter_methods as mf
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
def rumfor_AHSI(filepath, outputfolder, mf_type):
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return
    bands, radiance = sd.AHSI_data.get_calibrated_radiance(filepath, 2100, 2500)
    uas_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum.txt"
    interval_uas_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum_interval5000.txt"
    process_radiance(
        radiance, uas_path, interval_uas_path, filepath, outputfolder, mf_type
    )


# 单个EMIT文件处理
def rumfor_EMIT(filepath, outputfolder, mf_type):
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return
    emit_bands, radiance = ed.get_emit_bands_array(filepath, 2100, 2500)
    uas_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\EMIT_unit_absorption_spectrum.txt"
    interval_uas_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_unit_absorption_spectrum.txt"
    process_radiance(
        radiance, uas_path, interval_uas_path, filepath, outputfolder, mf_type
    )


# 通用的radiance处理函数
def process_radiance(
    radiance, uas_path, interval_uas_path, filepath, outputfolder, mf_type
):
    bands, uas = nf.open_unit_absorption_spectrum(uas_path, 2100, 2500)
    bands, interval_uas = nf.open_unit_absorption_spectrum(
        interval_uas_path, 2100, 2500
    )
    if mf_type == 0:
        enhancement = mf.matched_filter(
            radiance,
            uas,
            is_iterate=False,
            is_albedo=False,
            is_filter=False,
            is_columnwise=True,
        )
    elif mf_type == 1:
        enhancement = mf.modified_matched_filter(
            radiance,
            uas,
            interval_uas,
            is_iterate=False,
            is_albedo=False,
            is_filter=False,
            is_columnwise=True,
        )
    else:
        print("Invalid mf_type: 0 for original mf and 1 for modified mf")
        return
    if filepath.endswith(".tif"):
        ad.export_array_to_tiff(enhancement, filepath, outputfolder)
    else:
        ed.export_array_to_nc(enhancement, filepath, outputfolder)


if __name__ == "__main__":
    directory = "C:\\Users\\RS\Downloads\\EMITL1BRAD_001-20240822_062006"
    testpaths = file_paths = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".nc") and file.startswith("EMIT_L1B_RAD")
    ]
    outputfolder = "I:\\EMIT\\runfor1"
    for testpath in testpaths:
        print(f"{testpath} is processing.")
        rumfor_EMIT(testpath, outputfolder, 0)
