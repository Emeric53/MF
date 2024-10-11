import numpy as np
from matplotlib import pyplot as plt

import sys

sys.path.append("C://Users//RS//VSCode//matchedfiltermethod//src")
from utils.satellites_data import general_functions as gf


satellite_channels = "C://Users//RS//VSCode//matchedfiltermethod//src//data//satellite_channels//AHSI_channels.npz"


def sensitivity_analyse(filepath_folder: dict, low, high):
    base_radiance = None
    diff_list = []
    standardized_diff_list = []
    for file_path in filepath_folder:
        if base_radiance is None:
            wvls, base_radiance = gf.get_simulated_satellite_radiance(
                file_path, satellite_channels, low, high
            )
            continue
        # 2. 读取文件
        _, radiance_data = gf.get_simulated_satellite_radiance(
            file_path, satellite_channels, low, high
        )
        diff_list.append((radiance_data - base_radiance) / base_radiance)
        standardized_diff_list.append(
            max_min_standardization((radiance_data - base_radiance) / base_radiance)
        )

    return wvls, diff_list, standardized_diff_list


def max_min_standardization(data: np.ndarray):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def watervapor_sensitivity_analysis():
    filepath_folder = [
        f"C://PcModWin5//bin//batch//wv_{int(i)}_tape7.txt"
        for i in np.linspace(0, 5, 6)
    ]
    wvls, diff_list, standardized_diff_list = sensitivity_analyse(
        filepath_folder, 1500, 1800
    )
    fig, axes = plt.subplots(2, 1)
    for i in range(len(standardized_diff_list)):
        axes[0].plot(wvls, diff_list[i])
    for i in range(len(standardized_diff_list)):
        axes[1].plot(wvls, standardized_diff_list[i])
    plt.savefig(
        "C://Users//RS//VSCode//matchedfiltermethod//tasks//sensitivity_analysis//swir1_wv_sensitivity.png"
    )
    wvls, diff_list, standardized_diff_list = sensitivity_analyse(
        filepath_folder, 2100, 2500
    )
    fig, axes = plt.subplots(2, 1)
    for i in range(len(standardized_diff_list)):
        axes[0].plot(wvls, diff_list[i])
    for i in range(len(standardized_diff_list)):
        axes[1].plot(wvls, standardized_diff_list[i])
    plt.savefig(
        "C://Users//RS//VSCode//matchedfiltermethod//tasks//sensitivity_analysis//swir2_wv_sensitivity.png"
    )


def aerosol_sensitivity_analysis():
    filepath_folder = [
        f"C://PcModWin5//bin//batch//aerosol_{int(i)}_tape7.txt"
        for i in np.linspace(1, 4, 4)
    ]
    wvls, diff_list, standardized_diff_list = sensitivity_analyse(
        filepath_folder, 1500, 1800
    )
    fig, axes = plt.subplots(2, 1)
    for i in range(len(standardized_diff_list)):
        axes[0].plot(wvls, diff_list[i])
    for i in range(len(standardized_diff_list)):
        axes[1].plot(wvls, standardized_diff_list[i])
    plt.savefig(
        "C://Users//RS//VSCode//matchedfiltermethod//tasks//sensitivity_analysis//swir1_aerosol_sensitivity.png"
    )
    wvls, diff_list, standardized_diff_list = sensitivity_analyse(
        filepath_folder, 2100, 2500
    )
    fig, axes = plt.subplots(2, 1)
    for i in range(len(standardized_diff_list)):
        axes[0].plot(wvls, diff_list[i])
    for i in range(len(standardized_diff_list)):
        axes[1].plot(wvls, standardized_diff_list[i])
    plt.savefig(
        "C://Users//RS//VSCode//matchedfiltermethod//tasks//sensitivity_analysis//swir2_aerosol_sensitivity.png"
    )


def albedo_sensitivity_analysis():
    filepath_folder = [
        f"C://PcModWin5//bin//batch//albedo_{int(i)}_tape7.txt"
        for i in np.linspace(2, 6, 5)
    ]
    wvls, diff_list, standardized_diff_list = sensitivity_analyse(
        filepath_folder, 1500, 1800
    )

    fig, axes = plt.subplots(2, 1)
    for i in range(len(standardized_diff_list)):
        axes[0].plot(wvls, diff_list[i])
    for i in range(len(standardized_diff_list)):
        axes[1].plot(wvls, standardized_diff_list[i])
    plt.savefig(
        "C://Users//RS//VSCode//matchedfiltermethod//tasks//sensitivity_analysis//swir1_albedo_sensitivity.png"
    )
    wvls, diff_list, standardized_diff_list = sensitivity_analyse(
        filepath_folder, 2100, 2500
    )
    fig, axes = plt.subplots(2, 1)
    for i in range(len(standardized_diff_list)):
        axes[0].plot(wvls, diff_list[i])
    for i in range(len(standardized_diff_list)):
        axes[1].plot(wvls, standardized_diff_list[i])
    plt.savefig(
        "C://Users//RS//VSCode//matchedfiltermethod//tasks//sensitivity_analysis//swir2_albedo_sensitivity.png"
    )


if __name__ == "__main__":
    watervapor_sensitivity_analysis()
    aerosol_sensitivity_analysis()
    albedo_sensitivity_analysis()
