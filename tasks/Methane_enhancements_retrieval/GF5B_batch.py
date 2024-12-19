import pathlib as pl
import os
import sys
import re
import shutil
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from methane_retrieval_algorithms import columnwise_matchedfilter
from methane_retrieval_algorithms import matchedfilter
from scipy.ndimage import median_filter, gaussian_filter
from utils import satellites_data as sd
from utils import generate_radiance_lut_and_uas as glut


# 单个GF5B文件处理
def single_GF5B_run(filepath, outputfolder):
    low_wavelength = 2150
    high_wavelength = 2500
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    # rgb_outputfile = os.path.join(
    #     outputfolder, filename.replace("_SW.tif", "_RGB_corrected.tif")
    # )
    rgb_file = os.path.join(outputfolder, filename.replace("_SW.tif", "_RGB.tif"))
    if not os.path.exists(rgb_file):
        # os.remove(rgb_outputfile
        sd.GF5B_data.export_rgb_tiff(filepath, outputfolder)

    if os.path.exists(outputfile):
        # os.remove(outputfile)
        mask_file = os.path.join(
            outputfolder, filename.replace("_SW.tif", "final_mask.tif")
        )
        if not os.path.exists(mask_file):
            single_GF5B_plume_run(outputfile, outputfolder)
        return

    # # 基于 波段范围 读取辐射定标后的radiance的cube
    # _, AHSI_radiance = sd.GF5B_data.get_calibrated_radiance(
    #     filepath, low_wavelength, high_wavelength
    # )
    # # 读取 sza，地表高程的参数
    # sza, altitude = sd.GF5B_data.get_sza_altitude(filepath)
    # # 生成初始单位吸收谱 用于计算
    # _, uas = glut.generate_satellite_uas_for_specific_range_from_lut(
    #     "AHSI", 0, 50000, low_wavelength, high_wavelength, sza, altitude
    # )
    # try:
    #     enhancement = columnwise_matchedfilter.columnwise_matched_filter(
    #         AHSI_radiance, uas, True, True
    #     )

    #     output_rpb = outputfile.replace(".tif", ".rpb")
    #     if not os.path.exists(output_rpb):
    #         original_rpb = filepath.replace(".tif", ".rpb")
    #         shutil.copy(original_rpb, output_rpb)
    #     sd.GF5B_data.export_ahsi_array_to_tiff(
    #         enhancement, filepath, outputfolder, output_filename=None, orthorectify=True
    #     )
    # except Exception as e:
    #     print("Error in processing: ", filepath)
    #     print(e)


# 单个GF5B 烟羽处理
def single_GF5B_plume_run(filepath, outputfolder):
    filename = os.path.basename(filepath)
    basename = filename.replace("_SW.tif", "")

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    try:
        # 读取图像数据
        enhancement = sd.general_functions.read_tiff_in_numpy(filepath)[0, :, :]

        # 获取原始 rpb 文件路径
        original_rpb = filepath.replace(".tif", ".rpb")

        # 确保原始 rpb 文件存在
        if not os.path.exists(original_rpb):
            print(f"Error: Missing original RPB file for {filepath}")
            return

        # 1. Mask after significance test
        mean_value = np.mean(enhancement)
        std_dev = np.std(enhancement)
        significance_threshold = mean_value + 2 * std_dev
        mask_significance_test = enhancement > significance_threshold
        mask_significance_test_path = os.path.join(
            outputfolder, f"{basename}_mask_significance_test.tif"
        )

        # 创建临时 RPB 文件并导出正射校正后的掩膜
        temp_rpb_path = mask_significance_test_path.replace(".tif", ".rpb")
        print(temp_rpb_path)
        shutil.copy(original_rpb, temp_rpb_path)
        if os.path.exists(mask_significance_test_path):
            print("File already exists: ", mask_significance_test_path)
        sd.GF5B_data.export_ahsi_array_to_tiff(
            mask_significance_test.astype(np.uint8),
            filepath,
            outputfolder,
            output_filename=f"{basename}_mask_significance_test.tif",
            orthorectify=True,
        )

        # 2. Mask after median filter
        median_filtered_mask = (
            median_filter(mask_significance_test.astype(float), size=3) > 0.5
        )
        mask_median_filter_path = os.path.join(
            outputfolder, f"{basename}_mask_median_filter.tif"
        )

        # 创建临时 RPB 文件并导出正射校正后的掩膜
        temp_rpb_path = mask_median_filter_path.replace(".tif", ".rpb")
        shutil.copy(original_rpb, temp_rpb_path)
        sd.GF5B_data.export_ahsi_array_to_tiff(
            median_filtered_mask.astype(np.uint8),
            filepath,
            outputfolder,
            output_filename=f"{basename}_mask_median_filter.tif",
            orthorectify=True,
        )

        # 3. Mask after Gaussian filter
        gaussian_filtered_mask = gaussian_filter(
            median_filtered_mask.astype(float), sigma=1
        )
        mask_gaussian_filter_path = os.path.join(
            outputfolder, f"{basename}_mask_gaussian_filter.tif"
        )

        # 创建临时 RPB 文件并导出正射校正后的掩膜
        temp_rpb_path = mask_gaussian_filter_path.replace(".tif", ".rpb")
        shutil.copy(original_rpb, temp_rpb_path)
        sd.GF5B_data.export_ahsi_array_to_tiff(
            gaussian_filtered_mask,
            filepath,
            outputfolder,
            output_filename=f"{basename}_mask_gaussian_filter.tif",
            orthorectify=True,
        )

        # 4. Mask after thresholding
        final_mask = gaussian_filtered_mask > 0.5
        final_mask_path = os.path.join(outputfolder, f"{basename}_final_mask.tif")

        # 创建临时 RPB 文件并导出正射校正后的掩膜
        temp_rpb_path = final_mask_path.replace(".tif", ".rpb")
        shutil.copy(original_rpb, temp_rpb_path)
        sd.GF5B_data.export_ahsi_array_to_tiff(
            final_mask.astype(np.uint8),
            filepath,
            outputfolder,
            output_filename=f"{basename}_final_mask.tif",
            orthorectify=True,
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


def is_within_province_shapefile(data_name, province_shapefile):
    match = re.search(r"E([+-]?\d+\.\d+).*N([+-]?\d+\.\d+)", data_name)
    if match:
        longitude = float(match.group(1))
        latitude = float(match.group(2))
        point = Point(longitude, latitude)
        if province_shapefile.contains(point).any():
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


def batch_GF5B_run(filefolder_list, outputfolder, province_region_shapefile):
    for filefolder in filefolder_list:
        print("Current folder: ", filefolder)
        filepathlist = get_swir_filepath(filefolder)
        for filepath in filepathlist:
            if os.path.exists(filepath) is False:
                continue
            print("Current file: ", filepath)
            if (
                is_within_province_shapefile(filepath, province_region_shapefile)
                is True
            ):
                print(f"current file {filepath} is in targeted province")
                try:
                    single_GF5B_run(filepath, outputfolder)
                except Exception as e:
                    print("Error in singel run: ", filepath)
                    print(e)


def pickup_result(outputfolder, target_folder):
    # 源文件夹路径
    source_folder = outputfolder

    # 目标文件夹路径
    # target_folder = outputfolder + "_result"

    # 特定的文件扩展名（如以“.txt”结尾）
    file_extension = "_corrected.tif"  # 修改为您想要筛选的文件扩展名

    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的文件
    for filename in os.listdir(source_folder):
        # 构建完整的文件路径
        file_path = os.path.join(source_folder, filename)

        # 仅处理文件，忽略文件夹
        if os.path.isfile(file_path):
            # 检查文件是否以指定的扩展名结尾
            if filename.endswith(file_extension):
                # 构建目标文件路径
                target_file_path = os.path.join(target_folder, filename)

                # 移动文件到目标文件夹
                shutil.move(file_path, target_file_path)
                print(f"已移动文件: {filename}")


filepath = r"C:\Users\RS\Desktop\Lifei_essay_data\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985_SW.tif"
filepath = (
    r"C:\Users\RS\Desktop\hi\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985_SW.tif"
)
outputfolder = r"C:\Users\RS\Desktop\hi"

if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
# single_GF5B_plume_run(filepath, outputfolder)
# single_GF5B_run(filepath, outputfolder)

filefolder_list = [
    "I:\\AHSI_part2",
    "K:\\AHSI_part1",
    "M:\\AHSI_part3",
    "J:\\AHSI_part4",
]

# shanxi_region = [111.0, 114.5, 34.5, 40.8]
# outputfolder = r"L:\shanxi_gf5b"
# if not os.path.exists(outputfolder):
#     os.mkdir(outputfolder)
# pickup_result(outputfolder)
# batch_GF5B_run(filefolder_list, outputfolder, shanxi_region)

shanxi_region = [111.0, 114.5, 34.5, 40.8]
shanxi_region_shapefile = gpd.read_file(r"L:\行政区划\中国国省界SHP\shanxi.shp")
outputfolder = r"L:\shanxi_gf5b"
if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
target_folder = r"G:\shanxi_gf5b_result"
pickup_result(outputfolder, target_folder)
# batch_GF5B_run(filefolder_list, outputfolder, shanxi_region_shapefile)

xinjiang_region = [80.0, 95.0, 35.0, 49.0]
xinjiang_region_shapefile = gpd.read_file(r"L:\行政区划\中国国省界SHP\xinjiang.shp")
outputfolder = r"L:\xinjiang_gf5b"
if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)

target_folder = r"G:\xinjiang_gf5b_result"
pickup_result(outputfolder, target_folder)
# batch_GF5B_run(filefolder_list, outputfolder, xinjiang_region_shapefile)

# neimenggu_region = [97.0, 126.0, 37.0, 53.0]
# neimenggu_region_shapefile = gpd.read_file(r"L:\行政区划\中国国省界SHP\neimenggu.shp")
# outputfolder = r"L:\neimenggu_gf5b"
# if not os.path.exists(outputfolder):
#     os.mkdir(outputfolder)
# target_folder = r"G:\neimenggu_gf5b_result"
# pickup_result(outputfolder,target_folder)
# batch_GF5B_run(filefolder_list, outputfolder, neimenggu_region_shapefile)

world_region = [-180.0, 180.0, -90.0, 90.0]
# filefolder_list = ["C:\\Users\\RS\\Desktop\\Lifei_essay_data"]
# outputfolder = r"C:\Users\RS\Desktop\Lifei_essay_data\Lifei_essay_result"
# if not os.path.exists(outputfolder):
#     os.mkdir(outputfolder)
# batch_GF5B_run(filefolder_list,outputfolder, world_region)
