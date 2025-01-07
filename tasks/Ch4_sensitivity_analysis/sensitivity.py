import numpy as np
from matplotlib import pyplot as plt

import sys
import utils
# from utils.satellites_data import general_functions as gf


satellite_channels = "C://Users//RS//VSCode//matchedfiltermethod//src//data//satellite_channels//AHSI_channels.npz"


# def sensitivity_analyse(basepath, filepath_list: list, low, high):
#     base_radiance = None
#     diff_list = []
#     standardized_diff_list = []
#     wvls, base_radiance = wvls, base_radiance = gf.get_simulated_satellite_radiance(
#         basepath, satellite_channels, low, high
#     )
#     for file_path in filepath_list:
#         _, radiance_data = gf.get_simulated_satellite_radiance(
#             file_path, satellite_channels, low, high
#         )

#         diff_list.append((radiance_data - base_radiance) / base_radiance)
#         standardized_diff_list.append(
#             max_min_standardization((radiance_data - base_radiance) / base_radiance)
#         )
#     return wvls, diff_list, standardized_diff_list


# def max_min_standardization(data: np.ndarray):
#     return (data - np.min(data)) / (np.max(data) - np.min(data))


def show_radiance_graph(filepath, low, high):
    base_radiance = None
    diff_list = []
    standardized_diff_list = []
    wvls, base_radiance = gf.get_simulated_satellite_radiance(
        filepath, satellite_channels, low, high
    )
    fig, axes = plt.subplots(2, 1)
    # 假设我们想筛选波长在min_wvl和max_wvl之间的部分
    min_wvl = 2150  # 最小波长（例如）
    max_wvl = 2500  # 最大波长（例如）

    # 筛选出满足条件的波长索引
    selected_indices = (wvls >= min_wvl) & (wvls <= max_wvl)
    for i in range(len(standardized_diff_list)):
        axes[0].plot(wvls[selected_indices], diff_list[i][selected_indices])
    for i in range(len(standardized_diff_list)):
        axes[1].plot(
            wvls[selected_indices], standardized_diff_list[i][selected_indices]
        )
    plt.savefig(
        f"C://Users//RS//VSCode//matchedfiltermethod//tasks//sensitivity_analysis//swir2_{name}_sensitivity.png"
    )


def sensitivity_analyse(basepath, filepath_list: list, low, high):
    base_radiance = None
    diff_list = []
    standardized_diff_list = []
    # wvls, base_radiance = gf.get_simulated_satellite_radiance(
    #     basepath, satellite_channels, low, high
    # )

    wvls, base_radiance = gf.read_simulated_radiance(basepath)
    wvls, base_radiance = gf.slice_data(wvls, base_radiance, low, high)
    # 存储所有差异数据以计算全局最大最小值
    all_diffs = []

    for file_path in filepath_list:
        # _, radiance_data = gf.get_simulated_satellite_radiance(
        #     file_path, satellite_channels, low, high
        # )
        wvls, radiance_data = gf.read_simulated_radiance(file_path)
        wvls, radiance_data = gf.slice_data(wvls, radiance_data, low, high)
        # 计算差异并存储
        diff = ((radiance_data - base_radiance) / base_radiance) * 100
        diff_list.append(diff)
        all_diffs.append(diff)

    # 将所有差异平铺为一个数组来计算全局最小值和最大值
    all_diffs = np.concatenate(all_diffs, axis=0)
    global_min = np.min(all_diffs)
    global_max = np.max(all_diffs)

    # 基于全局最小值和最大值进行标准化
    for diff in diff_list:
        standardized_diff_list.append((diff - global_min) / (global_max - global_min))

    return wvls, diff_list, standardized_diff_list


def watervapor_sensitivity_analysis():
    # 读取不同水汽浓度的辐射数据 命名对应 0.5，1，2，3，4，5
    base_path = "C://PcModWin5//bin//batch//base_tape7.txt"
    filepath_list = [
        f"C://PcModWin5//bin//batch//wv_{int(i)}_tape7.txt"
        for i in np.linspace(1, 6, 6)
    ]
    wvls, diff_list, standardized_diff_list = sensitivity_analyse(
        base_path, filepath_list, 1500, 2500
    )

    export_graphes(wvls, diff_list, standardized_diff_list, "watervapor")


def aerosol_sensitivity_analysis():
    base_path = "C://PcModWin5//bin//batch//base_tape7.txt"
    # 1 for rural 23km 2 for marine 23km 3 for rural 5km 4 for urban 5km
    filepath_folder = [
        f"C://PcModWin5//bin//batch//aerosol_{int(i)}_tape7.txt"
        for i in np.linspace(1, 4, 4)
    ]
    wvls, diff_list, standardized_diff_list = sensitivity_analyse(
        base_path, filepath_folder, 1500, 2500
    )

    export_graphes(wvls, diff_list, standardized_diff_list, "aerosol")


def albedo_sensitivity_analysis():
    base_path = "C://PcModWin5//bin//batch//base_tape7.txt"
    filepath_folder = [
        f"C://PcModWin5//bin//batch//albedo_{int(i)}_tape7.txt"
        for i in np.linspace(20, 60, 5)
    ]
    wvls, diff_list, standardized_diff_list = sensitivity_analyse(
        base_path, filepath_folder, 1600, 1800
    )

    export_graphes(wvls, diff_list, standardized_diff_list, "albedo")


def co2_sensitivity_analysis():
    base_path = "C://PcModWin5//bin//batch//co2_420_tape7.txt"
    filepath_folder = [
        f"C://PcModWin5//bin//batch//co2_{int(i)}_tape7.txt"
        for i in np.linspace(430, 460, 4)
    ]
    wvls, diff_list, standardized_diff_list = sensitivity_analyse(
        base_path, filepath_folder, 1500, 2500
    )

    export_graphes(wvls, diff_list, standardized_diff_list, "co2")


def export_graphes(wvls, diff_list, standardized_diff_list, name):
    fig, axes = plt.subplots(2, 1)
    # 假设我们想筛选波长在min_wvl和max_wvl之间的部分
    min_wvl = 1500  # 最小波长（例如）
    max_wvl = 2500  # 最大波长（例如）

    # 筛选出满足条件的波长索引
    selected_indices = (wvls >= min_wvl) & (wvls <= max_wvl)
    for i in range(len(standardized_diff_list)):
        axes[0].plot(wvls[selected_indices], diff_list[i][selected_indices])
    for i in range(len(standardized_diff_list)):
        axes[1].plot(
            wvls[selected_indices], standardized_diff_list[i][selected_indices]
        )
    plt.savefig(
        f"C://Users//RS//VSCode//matchedfiltermethod//tasks//sensitivity_analysis//swir_{name}_sensitivity.png"
    )
    fig, axes = plt.subplots(2, 1)
    # 假设我们想筛选波长在min_wvl和max_wvl之间的部分
    min_wvl = 1600  # 最小波长（例如）
    max_wvl = 1850  # 最大波长（例如）

    # 筛选出满足条件的波长索引
    selected_indices = (wvls >= min_wvl) & (wvls <= max_wvl)
    for i in range(len(standardized_diff_list)):
        axes[0].plot(wvls[selected_indices], diff_list[i][selected_indices])
    for i in range(len(standardized_diff_list)):
        axes[1].plot(
            wvls[selected_indices], standardized_diff_list[i][selected_indices]
        )
    plt.savefig(
        f"C://Users//RS//VSCode//matchedfiltermethod//tasks//sensitivity_analysis//swir1_{name}_sensitivity.png"
    )
    fig, axes = plt.subplots(2, 1)
    # 假设我们想筛选波长在min_wvl和max_wvl之间的部分
    min_wvl = 2150  # 最小波长（例如）
    max_wvl = 2500  # 最大波长（例如）

    # 筛选出满足条件的波长索引
    selected_indices = (wvls >= min_wvl) & (wvls <= max_wvl)
    for i in range(len(standardized_diff_list)):
        axes[0].plot(wvls[selected_indices], diff_list[i][selected_indices])
    for i in range(len(standardized_diff_list)):
        axes[1].plot(
            wvls[selected_indices], standardized_diff_list[i][selected_indices]
        )
    plt.savefig(
        f"C://Users//RS//VSCode//matchedfiltermethod//tasks//sensitivity_analysis//swir2_{name}_sensitivity.png"
    )


if __name__ == "__main__":
    # watervapor_sensitivity_analysis()
    # aerosol_sensitivity_analysis()
    # albedo_sensitivity_analysis()
    pass
    # co2_sensitivity_analysis()
