import os
import re
import shutil

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from utils import satellites_data as sd
from utils import generate_radiance_lut_and_uas as glut
from methane_retrieval_algorithms import columnwise_matchedfilter


# 单个GF5B文件处理
def single_EMIT_run(filepath, outputfolder):
    low_wavelength = 2150
    high_wavelength = 2500
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename.replace(".nc", "_enhanced.tif"))
    # rgbfile = os.path.join(outputfolder, filename.replace(".nc", "_RGB.tif"))
    # if not os.path.exists(rgbfile):
    #     sd.EMIT_data.export_emit_rgb_array_to_tif(filepath, outputfolder)
    if os.path.exists(outputfile):
        return

    # 读取 radiance 数组
    _, EMIT_radiance = sd.EMIT_data.get_emit_bands_array(
        filepath, low_wavelength, high_wavelength
    )

    # 读取 sza，地表高程的参数
    sza, altitude = sd.EMIT_data.get_sza_altitude(filepath)

    # 生成初始单位吸收谱 用于计算
    _, uas = glut.generate_satellite_uas_for_specific_range_from_lut(
        "EMIT", 0, 50000, low_wavelength, high_wavelength, sza, altitude
    )

    try:
        enhancement = columnwise_matchedfilter.columnwise_matched_filter(
            EMIT_radiance, uas, True, True
        )
        # 将增强结果导出为 tif 文件
        sd.EMIT_data.export_emit_array_to_nc_tif(enhancement, filepath, outputfolder)
        # mask1, mask2, mask3, mask4 = EMIT_mask(enhancement)
        # mask1_file = os.path.join(outputfolder, filename.replace(".nc", "_mask1.tif"))
        # mask2_file = os.path.join(outputfolder, filename.replace(".nc", "_mask2.tif"))
        # mask3_file = os.path.join(outputfolder, filename.replace(".nc", "_mask3.tif"))
        # mask4_file = os.path.join(outputfolder, filename.replace(".nc", "_mask4.tif"))
        # sd.EMIT_data.export_emit_array_to_tif(mask1, filepath, mask1_file)
        # sd.EMIT_data.export_emit_array_to_tif(mask2, filepath, mask2_file)
        # sd.EMIT_data.export_emit_array_to_tif(mask3, filepath, mask3_file)
        # sd.EMIT_data.export_emit_array_to_tif(mask4, filepath, mask4_file)
    except Exception as e:
        print("Error in processing: ", filepath)
        print(e)


def EMIT_mask(enhancement):
    # 1. Mask after significance test
    mean_value = np.mean(enhancement)
    std_dev = np.std(enhancement)
    significance_threshold = mean_value + 1.5 * std_dev
    mask_significance_test = enhancement > significance_threshold
    # 2. Mask after median filter
    median_filtered_mask = (
        median_filter(mask_significance_test.astype(float), size=3) > 0.5
    )
    # 3. Mask after Gaussian filter
    gaussian_filtered_mask = gaussian_filter(
        median_filtered_mask.astype(float), sigma=1
    )
    # 4. Mask after thresholding
    final_mask = gaussian_filtered_mask > 0.5

    return (
        mask_significance_test,
        median_filtered_mask,
        gaussian_filtered_mask,
        final_mask,
    )


def batch_EMIT_run(outputfolder, province_region):
    # 结果文件夹
    filefolder = "/media/emeric/Leap/EMIT_L1B_RAD_shanxi"
    # 用于存储符合条件的文件路径
    filepathlist = []

    # 遍历文件夹
    for root, _, files in os.walk(filefolder):
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
            try:
                single_EMIT_run(filepath, outputfolder)
            except Exception as e:
                print("Error in processing: ", filepath)
                print(e)


def batch_Plume_EMIT_run(plume_folder, radiance_folder, outputfolder):
    # 查找并打印匹配的文件对
    matched_files = find_matching_radiance(plume_folder, radiance_folder)
    for plume, radiance in matched_files.items():
        print(f"Plume file: {plume} -> Radiance file: {radiance}")
        output_plume = os.path.join(
            outputfolder, os.path.basename(radiance).replace(".nc", "_plumeproduct.tif")
        )
        if os.path.exists(output_plume) is False:
            shutil.copyfile(plume, output_plume)
        single_EMIT_run(radiance, outputfolder)


def find_matching_radiance(plume_folder, radiance_folder):
    # 定义文件名中的时间戳匹配模式
    timestamp_pattern = re.compile(r"_(\d{8}T\d{6})_")

    # 获取 plume 文件列表
    plume_files = [f for f in os.listdir(plume_folder) if f.endswith(".tif")]

    # 获取 radiance 文件列表
    radiance_files = [
        f for f in os.listdir(radiance_folder) if f.endswith(".nc") and "_RAD_" in f
    ]

    # 创建 plume 文件和对应 radiance 文件的映射字典
    file_mapping = {}

    for plume_file in plume_files:
        # 从 plume 文件名中提取时间戳
        match = timestamp_pattern.search(plume_file)
        if match:
            timestamp = match.group(1)

            # 在 radiance 文件中查找匹配的文件
            matched_radiance = next(
                (rf for rf in radiance_files if timestamp in rf), None
            )
            plume_file = os.path.join(plume_folder, plume_file)
            if matched_radiance:
                # 存储匹配的 plume 和 radiance 文件
                file_mapping[plume_file] = os.path.join(
                    radiance_folder, matched_radiance
                )
            else:
                print(f"No matching radiance file found for {plume_file}")

    return file_mapping


if __name__ == "__main__":
    # # test single file
    # filepath = r"C:\\Users\\RS\\Desktop\\Lifei_essay_data\\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985\\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985_SW.tif"
    # outputfolder = r"C:\\Users\RS\\Desktop\\hi"
    # if not os.path.exists(outputfolder):
    #     os.mkdir(outputfolder)
    # # single_GF5B_run(filepath, outputfolder)

    shanxi_region = [111.0, 114.5, 34.5, 40.8]
    shanxi_region = [0, 180, 0, 90]
    # shanxi_region_shapefile = gpd.read_file(r"L:\行政区划\中国国省界SHP\shanxi.shp")
    outputfolder = r"/media/emeric/Documents/shanxi_emit"
    os.makedirs(outputfolder, exist_ok=True)
    batch_EMIT_run(outputfolder, shanxi_region)

    # xinjiang_region = [80.0, 95.0, 35.0, 49.0]
    # xinjiang_region_shapefile = gpd.read_file(r"L:\行政区划\中国国省界SHP\xinjiang.shp")
    # outputfolder = r"L:\xinjiang_emit"
    # if not os.path.exists(outputfolder):
    #     os.mkdir(outputfolder)
    # batch_EMIT_run(outputfolder, xinjiang_region)

    # neimenggu_region = [97.0, 126.0, 37.0, 53.0]
    # neimenggu_region_shapefile = gpd.read_file(r"L:\行政区划\中国国省界SHP\neimenggu.shp")
    # outputfolder = r"L:\neimenggu_emit"
    # if not os.path.exists(outputfolder):
    #     os.mkdir(outputfolder)
    # batch_EMIT_run(outputfolder, neimenggu_region)

    # plume_folder = r"M:\EMITL2BCH4PLM_001-20241108_114914"  # 替换为 plume 文件夹的实际路径
    # radiance_folder = r"J:\EMIT\L1B"  # 替换为 radiance 文件夹的实际路径
    # outputfolder = r"G:\EMIT_plume_result_1sigma"  # 替换为输出文件夹的实际路径
    # batch_Plume_EMIT_run(plume_folder, radiance_folder, outputfolder)
