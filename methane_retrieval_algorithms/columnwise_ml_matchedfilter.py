import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

from utils import satellites_data as sd
from utils import generate_radiance_lut_and_uas as glut
from utils import simulate_images as si
from methane_retrieval_algorithms import ml_matchedfilter as mlmf


# column为计算单元的 多层匹配滤波算法
def columnwise_ml_matched_filter(
    data_cube: np.ndarray,
    initial_unit_absorption_spectrum: np.ndarray,
    uas_list: np.ndarray,
    transmittance_list: np.ndarray,
    iterate: bool = False,
    albedoadjust: bool = False,
    sparsity: bool = False,
    group_size: int = 5,
    dynamic_adjust: bool = True,  # 新增动态调整标志
    threshold: float = 5000,  # 初始浓度增强阈值
    threshold_step: float = 5000,  # 阈值调整步长
    max_threshold: float = 50000,  # 最大浓度增强阈值
) -> np.ndarray:
    """Calculate the methane enhancement of the image data based on the original matched filter method.

    Args:
        data_cube (np.ndarray): 3D array representing the image data cube.
        unit_absorption_spectrum (np.ndarray): 1D array representing the unit absorption spectrum.
        iterate (bool): Flag indicating whether to perform iterative computation.
        albedoadjust (bool): Flag indicating whether to adjust for albedo.
        sparsity (bool): Flag for sparsity adjustment, not used here but can be implemented.
        group_size (int): The number of columns in each group to process together.

    Returns:
        np.ndarray: 2D array representing the concentration of methane.
    """

    # Initialize the concentration array, matching satellite data dimensions
    bands, rows, cols = data_cube.shape
    concentration = np.full((rows, cols), np.nan)  # Set default to NaN

    if bands != len(initial_unit_absorption_spectrum):
        raise ValueError(
            "The number of bands in the data cube must match the length of the unit absorption spectrum."
        )

    # Calculate the number of groups
    num_groups = cols // group_size
    remaining_cols = cols % group_size

    # Process each group of columns independently
    for group_idx in range(num_groups):
        # Define the range of columns in this group
        col_start = group_idx * group_size
        col_end = col_start + group_size
        columns_in_group = range(col_start, col_end)

        # Process these columns together
        current_group = data_cube[:, :, columns_in_group]
        concentration[:, col_start:col_end] = mlmf.ml_matched_filter(
            current_group,
            initial_unit_absorption_spectrum,
            uas_list,
            transmittance_list,
            iterate,
            albedoadjust,
            sparsity,
            dynamic_adjust,
            threshold,
            threshold_step,
            max_threshold,
        )

    # Handle the remaining columns that don't form a complete group
    if remaining_cols > 0:
        col_start = num_groups * group_size
        columns_in_group = range(col_start, col_start + remaining_cols)
        current_group = data_cube[:, :, columns_in_group]
        concentration[:, col_start:] = mlmf.ml_matched_filter(
            current_group,
            initial_unit_absorption_spectrum,
            uas_list,
            transmittance_list,
            iterate,
            albedoadjust,
            sparsity,
            dynamic_adjust,
            threshold,
            threshold_step,
            max_threshold,
        )

    return concentration


# 测试函数： 模拟影像测试
def columnwise_ml_matched_filter_simulated_image_test():
    # load the plume numpy array
    plume = np.load("data/simulated_plumes/gaussianplume_1000_2_stability_D.npy")
    # generate a simulated satelite image with methaen plums
    simulated_radiance_cube = si.simulate_satellite_images_with_plume(
        "AHSI", plume, 25, 0, 2150, 2500, 0.01
    )
    # get the corresponding unit absorption spectrum
    _, initial_uas = glut.generate_satellite_uas_for_specific_range_from_lut(
        "AHSI", 0, 50000, 2150, 2500, 25, 0
    )

    # count the time
    startime = time.time()
    # use MF to procedd the retrieval
    uas_list, transmittance_list = mlmf.generate_uas_transmittance_list("AHSI", 25, 0)
    methane_concentration = columnwise_ml_matched_filter(
        simulated_radiance_cube,
        initial_uas,
        uas_list,
        transmittance_list,
        False,
        False,
        False,
        5,
    )
    finish_time = time.time()
    print(f"running time: {finish_time - startime}")

    # 计算统计信息
    mean_concentration = np.mean(methane_concentration)
    std_concentration = np.std(methane_concentration)
    max_concentration = np.max(methane_concentration)
    min_concentration = np.min(methane_concentration)
    print(f"Mean: {mean_concentration:.2f} ppm")
    print(f"Std: {std_concentration:.2f} ppm")
    print(f"Max: {max_concentration:.2f} ppm")
    print(f"Min: {min_concentration:.2f} ppm")

    # 创建图形和子图
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # 子图1：甲烷浓度二维数组的可视化
    im = axes[0].imshow(methane_concentration, cmap="viridis", origin="lower")
    axes[0].set_title("Methane Concentration Enhancement (2D)")
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Y Coordinate")

    # 将 colorbar 移到下方
    cbar = fig.colorbar(
        im, ax=axes[0], orientation="horizontal", shrink=0.7, fraction=0.046, pad=0.04
    )
    cbar.set_label("Methane Concentration (ppm)")

    # 在第一个子图上添加统计信息
    stats_text = (
        f"Mean: {mean_concentration:.2f} ppmm\n"
        f"Std: {std_concentration:.2f} ppmm\n"
        f"Max: {max_concentration:.2f} ppmm\n"
        f"Min: {min_concentration:.2f} ppmm"
    )
    axes[0].text(
        1.05,
        0.5,
        stats_text,
        transform=axes[0].transAxes,
        fontsize=12,
        va="center",
        bbox=dict(facecolor="white", alpha=0.6),
    )
    plume_mask = plume < 100
    # 子图2：甲烷浓度分布的直方图和 KDE 图
    sns.histplot(
        methane_concentration[plume_mask].flatten(), bins=50, kde=True, ax=axes[1]
    )
    axes[1].set_title("Distribution of Methane Concentration")
    axes[1].set_xlabel("Methane Concentration (ppm)")
    axes[1].set_ylabel("Frequency")

    # 调整布局
    fig.tight_layout()
    # 显示图表
    plt.show()
    return


# 测试函数： 真实影像测试
def columnwise_ml_matched_filter_real_image_test(filepath, outputfolder):
    # 获取 文件名称
    filename = os.path.basename(filepath)
    # 设置 输出文件 名称
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return

    # 获取影像切片
    _, image_cube = sd.GF5B_data.get_calibrated_radiance(filepath, 2150, 2500)
    # 取整幅影像的 部分切片 进行测试
    image_sample_cube = image_cube[:, 500:700, 700:900]

    _, unit_absorption_spectrum = (
        glut.generate_satellite_uas_for_specific_range_from_lut(
            "AHSI", 0, 50000, 2150, 2500, 25, 0
        )
    )
    # 基于column的匹配滤波算法
    startime = time.time()
    methane_concentration = columnwise_ml_matched_filter(
        image_sample_cube, unit_absorption_spectrum, True, True, False, group_size=5
    )
    finish_time = time.time()
    print(f"running time: {finish_time - startime}")

    # sd.AHSI_data.export_ahsi_array_to_tiff(
    #     methane_concentration,
    #     filepath,
    #     outputfolder,
    # )

    # 计算统计信息
    mean_concentration = np.mean(methane_concentration)
    std_concentration = np.std(methane_concentration)
    max_concentration = np.max(methane_concentration)
    min_concentration = np.min(methane_concentration)
    print(f"Mean: {mean_concentration:.2f} ppm")
    print(f"Std: {std_concentration:.2f} ppm")
    print(f"Max: {max_concentration:.2f} ppm")
    print(f"Min: {min_concentration:.2f} ppm")

    # 创建图形和子图
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # 子图1：甲烷浓度二维数组的可视化
    im = axes[0].imshow(methane_concentration, cmap="viridis", origin="lower")
    axes[0].set_title("Methane Concentration Enhancement (2D)")
    axes[0].set_xlabel("X Coordinate")
    axes[0].set_ylabel("Y Coordinate")

    # 将 colorbar 移到下方
    cbar = fig.colorbar(
        im, ax=axes[0], orientation="horizontal", shrink=0.7, fraction=0.046, pad=0.04
    )
    cbar.set_label("Methane Concentration (ppm)")

    # 在第一个子图上添加统计信息
    stats_text = (
        f"Mean: {mean_concentration:.2f} ppm\n"
        f"Std: {std_concentration:.2f} ppm\n"
        f"Max: {max_concentration:.2f} ppm\n"
        f"Min: {min_concentration:.2f} ppm"
    )
    axes[0].text(
        1.05,
        0.5,
        stats_text,
        transform=axes[0].transAxes,
        fontsize=12,
        va="center",
        bbox=dict(facecolor="white", alpha=0.6),
    )

    # 子图2：甲烷浓度分布的直方图和 KDE 图
    sns.histplot(methane_concentration.flatten(), bins=50, kde=True, ax=axes[1])
    axes[1].set_title("Distribution of Methane Concentration")
    axes[1].set_xlabel("Methane Concentration (ppm)")
    axes[1].set_ylabel("Frequency")

    # 调整布局
    fig.tight_layout()
    # 显示图表
    plt.show()

    return


def main():
    # 模拟影像测试
    columnwise_ml_matched_filter_simulated_image_test()
    # 真实影像测试
    filepath = "C:\\Users\\RS\\Desktop\\Lifei_essay_data\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222_SW.tif"
    outputfolder = "C:/Users/RS\\Desktop\\Lifei_essay_data\\Lifei_essay_result\\"
    columnwise_ml_matched_filter_real_image_test(filepath, outputfolder)


if __name__ == "__main__":
    main()
