import numpy as np
from matplotlib import pyplot as plt

import time
import math

from methane_retrieval_algorithms.matchedfilter import matched_filter as mf
from methane_retrieval_algorithms.columnwise_matchedfilter import (
    columnwise_matched_filter as cmf,
)
from methane_retrieval_algorithms.ml_matchedfilter import ml_matched_filter as mlmf
from methane_retrieval_algorithms.columnwise_ml_matchedfilter import (
    columnwise_ml_matched_filter as cmlmf,
)
from utils import satellites_data as sd
from utils import generate_radiance_lut_and_uas as glut


def PRISMA_retrieval(filepath):
    # 读取PRISMA数据
    _, radiance_cube = sd.PRISMA_data.get_prisma_bands_array(filepath, 2150, 2500)
    # 读取PRISMA数据的SZA和高度
    sza, altitude = sd.PRISMA_data.get_SZA_altitude(filepath)
    if altitude > 5:
        altitude = 5
    # 生成PRISMA的单位吸收谱
    _, uas = glut.generate_satellite_uas_for_specific_range_from_lut(
        "PRISMA", 0, 50000, 2150, 2500, sza, altitude
    )

    # 原始匹配滤波算法结果
    mf_enhancement = mf(radiance_cube, uas, True, True, True)
    cmf_enhancement = cmf(radiance_cube, uas, True, True, True, 5)

    # 多层匹配滤波算法结果
    # mlmf_enhancement = mlmf(radiance_cube,uas,"PRISMA",True,True,True)

    # 定义函数计算95%分位数内的直方图和统计信息
    def plot_histogram_and_stats(array, label):
        # 展开数组并计算95%分位数范围
        data = array.flatten()
        lower = np.percentile(data, 2.5)
        upper = np.percentile(data, 97.5)

        # 过滤出95%分位数范围内的数据
        filtered_data = data[(data >= lower) & (data <= upper)]

        # 绘制直方图
        plt.hist(
            filtered_data,
            bins=100,
            alpha=0.5,
            label=f"{label} (95% range)",
            edgecolor="black",
        )

        # 计算统计信息
        mean = np.mean(filtered_data)
        std = np.std(filtered_data)
        print(f"{label} Stats:")
        print(
            f"Mean: {mean:.2f}, Std: {std:.2f}, Min: {filtered_data.min():.2f}, Max: {filtered_data.max():.2f}"
        )

    # # 绘制两个数组的直方图对比
    # plt.figure(figsize=(10, 6))

    # plot_histogram_and_stats(mf_enhancement, 'Array 1')
    # plot_histogram_and_stats(cmf_enhancement, 'Array 2')

    # # 设置图例与标题
    # plt.legend()
    # plt.title('95% Percentile Range Histogram Comparison')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.show()

    # 结果导出为 tiff
    mf_output = filepath.replace(".he5", "_mf.tif")
    sd.PRISMA_data.location_calibration(mf_enhancement, filepath, mf_output)
    cmf_output = filepath.replace(".he5", "_cmf.tif")
    sd.PRISMA_data.location_calibration(cmf_enhancement, filepath, cmf_output)
    # mlmf_output = filepath.replace(".tif","_mlmf.tif")
    # sd.PRISMA_data.location_calibration(cmf_enhancement,filepath,mlmf_output)
    # return mf_enhancement,cmf_enhancement,mlmf_enhancement


filepath = "/home/emeric/Documents/stanford/PRISMA/PRS_L1_STD_OFFL_20221027182300_20221027182304_0001.he5"
# filepath = "/home/emeric/Documents/stanford/PRISMA/PRS_L1_STD_OFFL_20221130180952_20221130180956_0001.he5"
# filepath = "J:\stanford\PRISMA\PRS_L1_STD_OFFL_20221130180952_20221130180956_0001.he5"
start_time = time.time()
PRISMA_retrieval(filepath)
finish_time = time.time()
print("Time cost: ", finish_time - start_time)
