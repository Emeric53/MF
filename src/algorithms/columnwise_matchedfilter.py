import numpy as np
import os
import time

import sys

sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src")

import utils.satellites_data as sd
import utils.satellites_data.general_functions as gf


# columnwise matched filter algorithm 逐列计算
def columnwise_matched_filter(
    data_cube: np.ndarray,
    unit_absorption_spectrum: np.ndarray,
    iterate: bool = False,
    albedoadjust: bool = False,
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
    # Initialize the concentration array, matching satellite data dimensions
    _, rows, cols = data_cube.shape
    concentration = np.full((rows, cols), np.nan)  # Set default to NaN

    # Process each column independently for concentration enhancement calculation
    for col_index in range(cols):
        current_column = data_cube[:, :, col_index]

        # Identify valid rows in this column (non-NaN rows only)
        valid_rows = ~np.isnan(current_column[0, :])
        count_not_nan = np.count_nonzero(valid_rows)

        # If there are no valid (non-NaN) rows, skip this column
        if count_not_nan == 0:
            continue

        # Calculate the background and target spectra for valid rows only
        background_spectrum = np.nanmean(current_column, axis=1)
        target_spectrum = background_spectrum * unit_absorption_spectrum
        radiancediff_with_bg = (
            current_column[:, valid_rows] - background_spectrum[:, None]
        )

        # Compute covariance matrix and its pseudo-inverse for valid rows
        d_covariance = radiancediff_with_bg
        covariance = np.einsum("ij,kj->ik", d_covariance, d_covariance) / count_not_nan
        covariance_inverse = np.linalg.pinv(covariance)

        # Adjust albedo if necessary
        albedo = np.ones(rows)
        if albedoadjust:
            albedo[valid_rows] = (
                current_column[:, valid_rows].T @ background_spectrum
            ) / (background_spectrum.T @ background_spectrum)
            albedo = np.nan_to_num(albedo, nan=1.0)  # Replace NaNs in albedo

        # Initial concentration calculation
        numerator = radiancediff_with_bg.T @ covariance_inverse @ target_spectrum
        denominator = albedo[valid_rows] * (
            target_spectrum.T @ covariance_inverse @ target_spectrum
        )
        conc = np.full(rows, np.nan)  # Default to NaN for all rows
        conc[valid_rows] = numerator / denominator

        # Iterative update if specified
        if iterate:
            for _ in range(5):
                # Update the background and target spectra for valid rows only
                background_spectrum = np.nanmean(
                    current_column[:, valid_rows]
                    - albedo[valid_rows] * conc[valid_rows] * target_spectrum[:, None],
                    axis=1,
                )
                target_spectrum = background_spectrum * unit_absorption_spectrum
                radiancediff_with_bg = (
                    current_column[:, valid_rows] - background_spectrum[:, None]
                )

                # Update covariance and concentration for valid rows
                d_covariance = current_column[:, valid_rows] - (
                    albedo[valid_rows] * conc[valid_rows] * target_spectrum[:, None]
                    + background_spectrum[:, None]
                )
                covariance = (
                    np.einsum("ij,kj->ik", d_covariance, d_covariance) / count_not_nan
                )
                covariance_inverse = np.linalg.pinv(covariance)

                numerator = (
                    radiancediff_with_bg.T @ covariance_inverse @ target_spectrum
                )
                denominator = albedo[valid_rows] * (
                    target_spectrum.T @ covariance_inverse @ target_spectrum
                )
                conc[valid_rows] = np.maximum(numerator / denominator, 0.0)

        # Assign the computed concentration to the respective column in the final output
        concentration[:, col_index] = conc

    return concentration


def columnwise_matched_filter_test():
    # filepath = "C:\\Users\\RS\\Desktop\\GF5-02_李飞论文所用数据\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222_SW.tif"
    filepath = "I:\AHSI_part2\GF5B_AHSI_E114.0_N30.3_20231026_011349_L10000410291\GF5B_AHSI_E114.0_N30.3_20231026_011349_L10000410291_SW.tif"
    outputfolder = "C:\\Users\\RS\\Desktop\\GF5-02_李飞论文所用数据\\mf_result\\"
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    if os.path.exists(outputfile):
        return

    _, image_cube = sd.GF5B_data.get_calibrated_radiance(filepath, 2150, 2500)
    image_sample_cube = image_cube
    unit_absoprtion_spectrum_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\uas_files\\AHSI_unit_absorption_spectrum.txt"

    _, unit_absoprtion_spectrum = gf.open_unit_absorption_spectrum(
        unit_absoprtion_spectrum_path, 2150, 2500
    )
    startime = time.time()
    methane_concentration = columnwise_matched_filter(
        data_cube=image_sample_cube,
        unit_absorption_spectrum=unit_absoprtion_spectrum,
        iterate=True,
        albedoadjust=True,
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

    # # 创建图形和子图
    # fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # # 子图1：甲烷浓度二维数组的可视化
    # im = axes[0].imshow(methane_concentration, cmap='viridis', origin='lower')
    # axes[0].set_title("Methane Concentration Enhancement (2D)")
    # axes[0].set_xlabel("X Coordinate")
    # axes[0].set_ylabel("Y Coordinate")

    # # 将 colorbar 移到下方
    # cbar = fig.colorbar(im, ax=axes[0], orientation='horizontal', shrink=0.7, fraction=0.046, pad=0.04)
    # cbar.set_label("Methane Concentration (ppm)")

    # # 在第一个子图上添加统计信息
    # stats_text = (f"Mean: {mean_concentration:.2f} ppm\n"
    #             f"Std: {std_concentration:.2f} ppm\n"
    #             f"Max: {max_concentration:.2f} ppm\n"
    #             f"Min: {min_concentration:.2f} ppm")
    # axes[0].text(1.05, 0.5, stats_text, transform=axes[0].transAxes,
    #             fontsize=12, va='center', bbox=dict(facecolor='white', alpha=0.6))

    # # 子图2：甲烷浓度分布的直方图和 KDE 图
    # sns.histplot(methane_concentration.flatten(), bins=50, kde=True, ax=axes[1])
    # axes[1].set_title("Distribution of Methane Concentration")
    # axes[1].set_xlabel("Methane Concentration (ppm)")
    # axes[1].set_ylabel("Frequency")

    # # 调整布局
    # fig.tight_layout()

    # # 显示图表
    # plt.show()

    return


if __name__ == "__main__":
    columnwise_matched_filter_test()
