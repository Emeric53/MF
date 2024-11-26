import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os


from utils import simulate_images as si
from utils import satellites_data as sd
from utils import generate_radiance_lut_and_uas as glut


def matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    iterate: bool = False,
    albedoadjust: bool = False,
    sparsity: bool = False,
) -> np.ndarray:
    """Calculate the methane enhancement of the image data based on the original matched filter method.

    Args:
        data_cube (np.ndarray): 3D array representing the image data cube.
        unit_absorption_spectrum (np.ndarray): 1D array representing the unit absorption spectrum.
        iterate (bool): Flag indicating whether to perform iterative computation.
        albedoadjust (bool): Flag indicating whether to adjust for albedo.

    Returns:
        np.ndarray: 2D array representing the concentration of methane.
    """
    # Ensure data_cube is a 3D array
    if data_cube.ndim == 2:
        data_cube = data_cube[np.newaxis, :, :]

    bands, rows, cols = data_cube.shape
    concentration = np.zeros((rows, cols))

    # Step 1: Calculate background spectrum and target spectrum
    background_spectrum = np.nanmean(
        data_cube, axis=(1, 2)
    )  # Mean across rows and cols
    target_spectrum = background_spectrum * unit_absorption_spectrum

    # Step 2: Calculate radiance difference (handling NaNs)
    radiancediff_with_background = data_cube - background_spectrum[:, None, None]

    # Step 3: Compute covariance matrix (avoid NaNs)
    d_covariance = radiancediff_with_background
    covariance = np.tensordot(d_covariance, d_covariance, axes=((1, 2), (1, 2))) / (
        rows * cols
    )
    covariance += np.eye(covariance.shape[0]) * 1e-6  # Regularization
    covariance_inverse = np.linalg.inv(covariance)

    # Step 4: Adjust albedo if needed
    albedo = np.ones((rows, cols))
    if albedoadjust:
        albedo = np.einsum("ijk,i->jk", data_cube, background_spectrum) / np.dot(
            background_spectrum, background_spectrum
        )
        albedo = np.nan_to_num(albedo, nan=1.0)

    # Step 5: Precompute common denominator
    common_denominator = target_spectrum.T @ covariance_inverse @ target_spectrum

    # Step 6: Compute concentration (vectorized)
    numerator = np.einsum(
        "ijk,i->jk",
        radiancediff_with_background,
        np.dot(covariance_inverse, target_spectrum),
    )
    concentration = numerator / (albedo * common_denominator)

    # Step 7: Handle iteration for more accurate concentration calculation
    if iterate:
        l1filter = np.zeros((rows, cols))
        for _ in range(5):
            # Step 7.0: Handle sparsity
            if sparsity:
                l1filter = 1 / (concentration + 1e-6)

            # Step 7.1: Update residual (background and target spectra)
            residual = (
                data_cube
                - (albedo * concentration)[None, :, :] * target_spectrum[:, None, None]
            )

            background_spectrum = np.nanmean(residual, axis=(1, 2))

            target_spectrum = background_spectrum * unit_absorption_spectrum

            # Step 7.2: Recompute radiance difference and covariance
            radiancediff_with_background = (
                data_cube - background_spectrum[:, None, None]
            )

            d_covariance = (
                radiancediff_with_background
                - (albedo * concentration)[None, :, :] * target_spectrum[:, None, None]
            )
            covariance = np.tensordot(
                d_covariance, d_covariance, axes=((1, 2), (1, 2))
            ) / (rows * cols)
            covariance += np.eye(covariance.shape[0]) * 1e-6  # Regularization
            covariance_inverse = np.linalg.inv(covariance)

            # Step 7.3: Update common denominator and compute concentration
            common_denominator = (
                target_spectrum.T @ covariance_inverse @ target_spectrum
            )
            numerator = (
                np.einsum(
                    "ijk,i->jk",
                    radiancediff_with_background,
                    np.dot(covariance_inverse, target_spectrum),
                )
                - l1filter
            )
            concentration = np.maximum(numerator / (albedo * common_denominator), 0)

    return concentration


def matched_filter_simulated_image_test():
    # load the plume numpy array
    plume = np.load("data/simulated_plumes/gaussianplume_1000_2_stability_D.npy")
    # generate a simulated satelite image with methaen plums
    simulated_radiance_cube = si.simulate_satellite_images_with_plume(
        "AHSI", plume, 25, 0, 2150, 2500, 0.01
    )
    # get the corresponding unit absorption spectrum
    _, uas = glut.generate_satellite_uas_for_specific_range_from_lut(
        "AHSI", 0, 50000, 2150, 2500, 25, 0
    )
    # count the time
    startime = time.time()
    # use MF to procedd the retrieval
    methane_concentration = matched_filter(
        simulated_radiance_cube, uas, False, False, False
    )
    finish_time = time.time()
    print(f"running time: {finish_time - startime}")

    # 计算统计信息
    mean_concentration = np.mean(methane_concentration)
    std_concentration = np.std(methane_concentration)
    max_concentration = np.max(methane_concentration)
    min_concentration = np.min(methane_concentration)
    print(f"Mean: {mean_concentration:.2f} ppmm")
    print(f"Std: {std_concentration:.2f} ppmm")
    print(f"Max: {max_concentration:.2f} ppmm")
    print(f"Min: {min_concentration:.2f} ppmm")

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
    print(plume.max())
    print(methane_concentration.max())
    return


def matched_filter_real_image_test(filepath, outputfolder):
    # 获取文件名称
    filename = os.path.basename(filepath)
    # 设置输出文件路径
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return
    # 获取影像切片进行实验
    _, image_cube = sd.GF5B_data.get_calibrated_radiance(filepath, 2150, 2500)
    # 取整幅影像的 100*100 切片进行测试
    image_sample_cube = image_cube[:, 500:700, 700:900]
    _, unit_absorption_spectrum = (
        glut.generate_satellite_uas_for_specific_range_from_lut(
            "AHSI", 0, 50000, 2150, 2500, 25, 0
        )
    )
    # 使用匹配滤波算法进行实验
    startime = time.time()
    methane_concentration = matched_filter(
        image_sample_cube, unit_absorption_spectrum, True, True, False
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


if __name__ == "__main__":
    # 模拟影像测试
    matched_filter_simulated_image_test()
    # 真实影像测试
    # filepath = "C:\\Users\\RS\\Desktop\\Lifei_essay_data\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222_SW.tif"
    # outputfolder = "C:/Users/RS\\Desktop\\Lifei_essay_data\\Lifei_essay_result\\"
    # matched_filter_real_image_test()
