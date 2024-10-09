import numpy as np
from matplotlib import pyplot as plt

import sys

sys.path.append("C://Users//RS//VSCode//matchedfiltermethod//src")
from utils.satellites_data import general_functions as gf

filepath_folder = [
    f"C://PcModWin5//bin//batch//{int(i)}_0_0_tape7.txt"
    for i in np.linspace(0, 50000, 101)
]
satellite_channels = "C://Users//RS//VSCode//matchedfiltermethod//src//data//satellite_channels//AHSI_channels.npz"


def sensitivity_analyse(filepath_folder: dict, low, high):
    # 1. 获取模拟卫星辐射数据
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


# def sensitivity_analyse(filepath_folder: list, low, high):
#     # 1. 获取模拟卫星辐射数据
#     base_radiance = None
#     diff_list = []
#     for file_path in filepath_folder:
#         if base_radiance is None:
#             # 获取初始辐射值作为基准
#             wvls, base_radiance = gf.get_simulated_satellite_radiance(
#                 file_path, satellite_channels, low, high
#             )
#             continue
#         # 2. 读取文件
#         _, radiance_data = gf.get_simulated_satellite_radiance(
#             file_path, satellite_channels, low, high
#         )
#         # 计算相对差异
#         diff = (radiance_data - base_radiance) / base_radiance
#         diff_list.append(diff)

#     # 3. 计算全局最大最小值，用于所有差异的归一化
#     all_diffs = np.concatenate(diff_list, axis=0)
#     global_min = np.min(all_diffs)
#     global_max = np.max(all_diffs)

#     # 4. 对所有差异进行统一归一化
#     standardized_diff_list = [
#         max_min_standardization(diff, global_min, global_max) for diff in diff_list
#     ]

#     return wvls, diff_list, standardized_diff_list


# def max_min_standardization(data: np.ndarray, global_min: float, global_max: float):
#     """根据全局最小值和最大值对数据进行标准化"""
#     return (data - global_min) / (global_max - global_min)


def wv_sensitivity_analysis():
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
    wv_sensitivity_analysis()
