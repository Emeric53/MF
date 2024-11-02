from osgeo import gdal
from matplotlib import pyplot as plt
import numpy as np

import pathlib as pl
import os
import sys

sys.path.append("c:\\Users\\RS\\VSCode\\matchedfiltermethod\src")
from utils.satellites_data.general_functions import (
    save_ndarray_to_tiff,
    read_tiff_in_numpy,
)


# ! 对ZY1 上的AHSI数据读取SZA和地面高程
def get_sza_altitude(filepath: str):
    hdr_file = filepath.replace(".dat", ".hdr")
    with open(hdr_file, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=")
                if key.strip() == "solar zenith":
                    sza = float(value.strip())
                    print(f"SZA: {sza}")
    return sza, 0


# 获取ahsi的波段信息
def get_ZY1_ahsi_bands():
    """
    get bands list of ahsi
    :param band_file:  filepath containing bands wavelength
    :return: bands list
    """
    # 读取校准文件
    wavelengths = np.load(
        "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\satellite_channels\\ZY1_channels.npz"
    )["central_wvls"]
    return wavelengths


def get_AHSI_radiances_from_dat(dat_file, low, high):
    radiance_cube = gdal.Open(dat_file)
    radiance = radiance_cube.ReadAsArray()
    wvls = get_ZY1_ahsi_bands()
    indices = np.where((wvls >= low) & (wvls <= high))[0]
    radiance = radiance[indices, :, :]
    return wvls[indices], radiance


def extract_wavelengths_from_hdr(hdr_file):
    wavelengths = []
    inside_wavelength_section = False

    # 打开并逐行读取 .hdr 文件
    with open(hdr_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("wavelength"):
                inside_wavelength_section = True
                continue
            if inside_wavelength_section:
                if line.startswith("{"):
                    # 波长数据的开始
                    continue
                elif line.startswith("}"):
                    # 波长数据的结束
                    break
                else:
                    # 添加波长值
                    wavelengths.extend([float(x) for x in line.split(",")])

    return np.array(wavelengths)


if __name__ == "__main__":
    # a = r"I:\\AHSI_part4\GF5B_AHSI_E83.9_N43.1_20230929_010957_L10000398404\GF5B_AHSI_E83.9_N43.1_20230929_010957_L10000398404_SW.tif"
    # bands, radiance = get_calibrated_radiance(a, 1500, 2300)
    # 使用示例
    hdr_file = "I:\stanford_campaign\Stanford_Campaign_GF5-02-AHSI\GF5B_AHSI_W112.1_N32.8_20221115_006332_L10000239663_VNSW_Rad.hdr"
    # metadata = read_hdr_file(hdr_file)
    dat_file = "I:\stanford_campaign\Stanford_Campaign_GF5-02-AHSI\GF5B_AHSI_W112.1_N32.8_20221115_006332_L10000239663_VNSW_Rad.dat"

    # wvls = extract_wavelengths_from_hdr(hdr_file)
    # print(wvls)
    # wavelength = get_ahsi_bands()
    # print(wavelength.shape)
