import numpy as np
from osgeo import gdal

import os

from utils.satellites_data.general_functions import (
    get_simulated_satellite_radiance,
)
from utils.generate_radiance_lut_and_uas import (
    batch_get_radiance_from_lut,
    get_satellite_radiance_spectrum_from_lut,
)
import utils.satellites_data as sd
import utils.satellites_data.general_functions as gf


# ! 模拟带甲烷烟羽的卫星遥感影像 大小和 烟羽数组大小一致
def simulate_satellite_images(
    radiance_path: str,
    satellite_name: str,
    image_shape,
    plume: np.ndarray,
    lower_wavelength: float = 2150,
    upper_wavelength: float = 2500,
    noise_level=0.01,
):
    """
    Simulate a radiance image with added plume effects and Gaussian noise.

    Parameters:
    radiance_path (str): Path to the file containing simulated radiance data.
    plume (array-like): Data representing the plume to be simulated.
    scaling_factor (float, optional): Scaling factor for radiance values. Default is 1.
    lower_wavelength (int, optional): Lower bound of the wavelength range in nm. Default is 2150.
    upper_wavelength (int, optional): Upper bound of the wavelength range in nm. Default is 2500.
    row_num (int, optional): Number of rows in the simulated image. Default is 100.
    col_num (int, optional): Number of columns in the simulated image. Default is 100.
    noise_level (float, optional): Standard deviation of the Gaussian noise relative to the signal. Default is 0.005.

    Returns:
    numpy.ndarray: Simulated radiance image with added plume effects and Gaussian noise.

    The function performs the following steps:
    1. Loads the simulated radiance spectrum from the specified path.
    2. Sets the shape of the image to be simulated.
    3. Generates a universal radiance cube image.
    4. Adds the plume effects to the radiance image.
    5. Adds Gaussian noise to the image.
    """
    # Load the simulated emit radiance spectrum
    channels_path = f"/home/emeric/Documents/GitHub/MF/data/satellite_channels/{satellite_name}_channels.npz"
    if not os.path.exists(channels_path):
        print("The satellite name is not supported.")
        return None

    bands, simulated_convolved_spectrum = get_simulated_satellite_radiance(
        radiance_path, channels_path, lower_wavelength, upper_wavelength
    )
    bands, simulated_radiance = get_satellite_radiance_spectrum_from_lut(
        satellite_name, 0, sza, 0, lower_wavelength, upper_wavelength
    )
    # Set the shape of the image that want to simulate
    band_num = len(bands)

    # Generate the universal radiance cube image
    simulated_image = simulated_convolved_spectrum.reshape(
        band_num, 1, 1
    ) * np.oneslike(plume)

    image_with_plume = simulated_image
    simulated_noisy_image = np.zeros_like(simulated_image)
    for i in range(band_num):  # Traverse each band
        current = simulated_convolved_spectrum[i]
        noise = np.random.normal(
            0, current * noise_level, (plume.shape[0], plume.shape[1])
        )  # Generate Gaussian noise
        simulated_noisy_image[i, :, :] = (
            image_with_plume[i, :, :] + noise
        )  # Add noise to the original data

    return simulated_noisy_image


# ! 模拟 不带 甲烷烟羽的卫星遥感影像
def simulate_satellite_images_without_plume(
    satellite_name: str,
    image_shape,
    sza,
    altitude,
    lower_wavelength: float = 2150,
    upper_wavelength: float = 2500,
    noise_level=0.01,
):
    """
    Simulate a radiance image with added plume effects and Gaussian noise.

    Parameters:
    radiance_path (str): Path to the file containing simulated radiance data.
    plume (array-like): Data representing the plume to be simulated.
    scaling_factor (float, optional): Scaling factor for radiance values. Default is 1.
    lower_wavelength (int, optional): Lower bound of the wavelength range in nm. Default is 2150.
    upper_wavelength (int, optional): Upper bound of the wavelength range in nm. Default is 2500.
    row_num (int, optional): Number of rows in the simulated image. Default is 100.
    col_num (int, optional): Number of columns in the simulated image. Default is 100.
    noise_level (float, optional): Standard deviation of the Gaussian noise relative to the signal. Default is 0.005.

    Returns:
    numpy.ndarray: Simulated radiance image with added plume effects and Gaussian noise.

    The function performs the following steps:
    1. Loads the simulated radiance spectrum from the specified path.
    2. Sets the shape of the image to be simulated.
    3. Generates a universal radiance cube image.
    4. Adds the plume effects to the radiance image.
    5. Adds Gaussian noise to the image.
    """

    row, col = image_shape[0], image_shape[1]

    wvls, radiance = get_satellite_radiance_spectrum_from_lut(
        satellite_name, 0, sza, altitude, lower_wavelength, upper_wavelength
    )

    radiance_cube = np.ones((row, col))[None, :, :] * radiance[:, None, None]

    # 噪声的标准差与 radiance_cube 的值成正比
    noise_stddev = radiance_cube * noise_level

    # 根据均值为 0 和标准差为 noise_stddev 生成高斯噪声
    noise = np.random.normal(0, noise_stddev)

    # 将噪声添加到辐射数据上
    noisy_radiance_cube = radiance_cube + noise
    return noisy_radiance_cube


# ! 模拟 带 甲烷烟羽的卫星遥感影像 大小和 烟羽数组大小一致
def simulate_satellite_images_with_plume(
    satellite_name: str,
    image_shape,
    plume: np.ndarray,
    sza,
    altitude,
    lower_wavelength: float = 2150,
    upper_wavelength: float = 2500,
    noise_level=0.01,
):
    """
    Simulate a radiance image with added plume effects and Gaussian noise.

    Parameters:
    radiance_path (str): Path to the file containing simulated radiance data.
    plume (array-like): Data representing the plume to be simulated.
    scaling_factor (float, optional): Scaling factor for radiance values. Default is 1.
    lower_wavelength (int, optional): Lower bound of the wavelength range in nm. Default is 2150.
    upper_wavelength (int, optional): Upper bound of the wavelength range in nm. Default is 2500.
    row_num (int, optional): Number of rows in the simulated image. Default is 100.
    col_num (int, optional): Number of columns in the simulated image. Default is 100.
    noise_level (float, optional): Standard deviation of the Gaussian noise relative to the signal. Default is 0.005.

    Returns:
    numpy.ndarray: Simulated radiance image with added plume effects and Gaussian noise.

    The function performs the following steps:
    1. Loads the simulated radiance spectrum from the specified path.
    2. Sets the shape of the image to be simulated.
    3. Generates a universal radiance cube image.
    4. Adds the plume effects to the radiance image.
    5. Adds Gaussian noise to the image.
    """

    # 获取模拟影像的行列数
    row, col = image_shape[0], image_shape[1]
    plume_row, plume_col = plume.shape

    # 检查模拟影像是否大于烟羽
    if row < plume_row or col < plume_col:
        raise ValueError("模拟影像尺寸小于烟羽尺寸，无法放置烟羽在中心位置！")

    # 构建一个模拟影像尺寸的零数组
    simulated_image = np.zeros((row, col))

    # 计算烟羽数组在模拟影像中放置的位置
    start_row = (row - plume_row) // 2
    start_col = (col - plume_col) // 2

    # 将烟羽数组放置在中心位置
    simulated_image[
        start_row : start_row + plume_row, start_col : start_col + plume_col
    ] = plume
    simulated_image_flat = simulated_image.flatten()

    wvls, radiance_cube = batch_get_radiance_from_lut(
        satellite_name, simulated_image_flat, sza, altitude
    )
    wvl_condition = np.logical_and(wvls >= lower_wavelength, wvls <= upper_wavelength)

    radiance_cube = radiance_cube.transpose(1, 0)
    radiance_cube = radiance_cube.reshape(radiance_cube.shape[0], row, col)[
        wvl_condition
    ]
    # 噪声的标准差与 radiance_cube 的值成正比
    noise_stddev = radiance_cube * noise_level

    # 根据均值为 0 和标准差为 noise_stddev 生成高斯噪声
    noise = np.random.normal(0, noise_stddev)

    # 将噪声添加到辐射数据上
    noisy_radiance_cube = radiance_cube + noise
    return simulated_image, noisy_radiance_cube


# ! 基于甲烷烟羽 构建 透射率cube，叠加在真实影像上
def add_plume_to_real_image(
    satellite_name: str,
    image_path: str,
    image_shape,
    plume: np.ndarray,
    lower_wavelength: float = 2150,
    upper_wavelength: float = 2500,
    output_path: str = None,
):
    if satellite_name == "AHSI":
        _, real_image = sd.GF5B_data.get_calibrated_radiance(
            image_path, lower_wavelength, upper_wavelength
        )
        sza, altitude = sd.GF5B_data.get_sza_altitude(image_path)

        row, col = image_shape[0], image_shape[1]
        plume_row, plume_col = plume.shape

        # 检查模拟影像是否大于烟羽
        if row < plume_row or col < plume_col:
            raise ValueError("模拟影像尺寸小于烟羽尺寸，无法放置烟羽在中心位置！")

        # 构建一个模拟影像尺寸的零数组
        simulated_image = np.zeros((row, col))

        # 计算烟羽数组在模拟影像中放置的位置
        start_row = (row - plume_row) // 2
        start_col = (col - plume_col) // 2

        # 将烟羽数组放置在中心位置
        simulated_image[
            start_row : start_row + plume_row, start_col : start_col + plume_col
        ] = plume
        simulated_image_flat = simulated_image.flatten()

        wvls, radiance_cube = batch_get_radiance_from_lut(
            satellite_name,
            simulated_image_flat,
            sza,
            altitude,
            lower_wavelength,
            upper_wavelength,
        )
        _, base_radiance = get_satellite_radiance_spectrum_from_lut(
            satellite_name, 0, 0, 0, lower_wavelength, upper_wavelength
        )

        transmittance_cube = np.transpose(radiance_cube / base_radiance[None, :])
        transmittance_cube = transmittance_cube.reshape(
            transmittance_cube.shape[0], row, col
        )

        real_image_clip = real_image[:, 200 : 200 + row, 200 : 200 + col]

        plume_added_image = real_image_clip * transmittance_cube
        if output_path is not None:
            gf.save_ndarray_to_tiff(plume_added_image, output_path)
            np.save(output_path.replace(".tif", ".npy"), plume_added_image)
        return simulated_image, plume_added_image


if __name__ == "__main__":
    # gf5b 切片 模拟 叠加影像
    image_file = r"/media/emeric/Bird/AHSI_part1/GF5B_AHSI_E111.6_N35.8_20231031_011422_L10000412729/GF5B_AHSI_E111.6_N35.8_20231031_011422_L10000412729_SW.tif"
    # 导出rgb影像
    # row, col = (200, 200)
    # outputpath = r"/home/emeric/Documents/GitHub/MF/data/simulated_images/GF5B_rgb.tif"
    # filepath = image_file
    # calibration_filepath = os.path.dirname(filepath) + "//GF5B_AHSI_RadCal_VNIR.raw"
    # ahsi_array = sd.GF5B_data.get_ahsi_array(filepath.replace("_SW.tif", "_VN.tif"))
    # coeffs = sd.GF5B_data.get_radiometric_calibration_coefficients(calibration_filepath)
    # calibrated_radiance = sd.GF5B_data.radiance_calibration(ahsi_array, coeffs)
    # red = calibrated_radiance[59, 200 : 200 + row, 200 : 200 + col]  # band 60
    # green = calibrated_radiance[38, 200 : 200 + row, 200 : 200 + col]  # band 39
    # blue = calibrated_radiance[19, 200 : 200 + row, 200 : 200 + col]  # band 20
    # rgb_radiance = np.stack((red, green, blue), axis=0)
    # sd.GF5B_data.save_ndarray_to_tiff(rgb_radiance, outputpath)

    # 导出GF5B 2300nm 波段
    # row, col = (100, 100)
    # outputpath = (
    #     r"/home/emeric/Documents/GitHub/MF/data/simulated_images/GF5B_2300nm.tif"
    # )
    # filepath = image_file
    # _, calibrated_radiance = sd.GF5B_data.get_calibrated_radiance(filepath, 2295, 2305)

    # red = calibrated_radiance[0, 200 : 200 + row, 200 : 200 + col]
    # np.save("data/simulated_images/GF5B_2300nm.npy", red)
    # sd.GF5B_data.save_ndarray_to_tiff(red, outputpath)

    # emit_plume_path = "/home/emeric/Documents/GitHub/MF/data/EMIT_plumes/EMIT_L2B_CH4PLM_001_20240610T011426_003234.tif"
    # plume = gdal.Open(emit_plume_path).ReadAsArray()
    # plume[plume < 100] = 0
    # emit_plume = plume
    # np.save("data/simulated_images/emitplume2.npy", emit_plume)
    # for i in [2, 4, 6, 8, 10]:
    #     # 加载高斯烟羽
    #     gaussian_plume = np.load(
    #         f"data/simulated_plumes/gaussianplume_1000_{i}_stability_D.npy"
    #     )

    #     # 模拟带烟羽的真实影像
    #     gaussian_plume, _ = add_plume_to_real_image(
    #         "AHSI",
    #         image_file,
    #         (100, 100),
    #         gaussian_plume,
    #         2150,
    #         2500,
    #         output_path=f"data/simulated_images/GF5B_gaussianplume_1000_{i}_D.tif",
    #     )
    #     # 模拟带烟羽的模拟影像
    #     _, gaussian_plume_cube = simulate_satellite_images_with_plume(
    #         "AHSI",
    #         (100, 100),
    #         gaussian_plume,
    #         25,
    #         0,
    #         2150,
    #         2500,
    #         0.01,
    #     )
    #     np.save(
    #         f"data/simulated_images/simulated_gaussianplume_1000_{i}_D.npy",
    #         gaussian_plume_cube,
    #     )
    #     np.save(f"data/simulated_images/gaussianplume_1000_{i}_D.npy", gaussian_plume)

    # # 加载emit烟羽
    # original_emit_plume = np.load("data/simulated_images/emitplume.npy")
    # for i in [1, 2, 3, 4, 5]:
    #     emit_plume = i * original_emit_plume
    #     # 模拟带烟羽的真实影像
    #     emit_plume, _ = add_plume_to_real_image(
    #         "AHSI",
    #         image_file,
    #         (100, 100),
    #         emit_plume,
    #         2150,
    #         2500,
    #         output_path=f"data/simulated_images/GF5B_emitplume1_{i}.tif",
    #     )
    #     np.save(f"data/simulated_images/emitplume1_{i}.npy", emit_plume)
    #     # 模拟带烟羽的模拟影像
    #     _, emitplume_cube = simulate_satellite_images_with_plume(
    #         "AHSI",
    #         (100, 100),
    #         emit_plume,
    #         25,original_emit_plume = np.load("data/simulated_images/emitplume2.npy")
    # for i in [1, 2, 3, 4, 5]:
    #     emit_plume = i * original_emit_plume
    #     # 模拟带烟羽的真实影像
    #     emit_plume, _ = add_plume_to_real_image(
    #         "AHSI",
    #         image_file,
    #         (100, 100),
    #         emit_plume,
    #         2150,
    #         2500,
    #         output_path=f"data/simulated_images/GF5B_emitplume2_{i}.tif",
    #     )
    #     np.save(f"data/simulated_images/emitplume2_{i}.npy", emit_plume)
    #     # 模拟带烟羽的模拟影像
    #     _, emitplume_cube = simulate_satellite_images_with_plume(
    #         "AHSI",
    #         (100, 100),
    #         emit_plume,
    #         25,
    #         0,
    #         2150,
    #         2500,
    #         0.01,
    #     )

    #     np.save(f"data/simulated_images/simulated_emitplume2_{i}.npy", emitplume_cube)

    #         0,
    #         2150,
    #         2500,
    #         0.01,
    #     )

    #     np.save(f"data/simulated_images/simulated_emitplume1_{i}.npy", emitplume_cube)

    # 加载emit烟羽
    # original_emit_plume = np.load("data/simulated_images/emitplume2.npy")
    # for i in [1, 2, 3, 4, 5]:
    #     emit_plume = i * original_emit_plume
    #     # 模拟带烟羽的真实影像
    #     emit_plume, _ = add_plume_to_real_image(
    #         "AHSI",
    #         image_file,
    #         (100, 100),
    #         emit_plume,
    #         2150,
    #         2500,
    #         output_path=f"data/simulated_images/GF5B_emitplume2_{i}.tif",
    #     )
    #     np.save(f"data/simulated_images/emitplume2_{i}.npy", emit_plume)
    #     # 模拟带烟羽的模拟影像
    #     _, emitplume_cube = simulate_satellite_images_with_plume(
    #         "AHSI",
    #         (100, 100),
    #         emit_plume,
    #         25,
    #         0,
    #         2150,
    #         2500,
    #         0.01,
    #     )

    #     np.save(f"data/simulated_images/simulated_emitplume2_{i}.npy", emitplume_cube)

    # # 模拟带烟羽的模拟影像
    # _, gaussian_plume_cube = simulate_satellite_images_with_plume(
    #     "AHSI",
    #     (100, 100),
    #     gaussian_plume,
    #     25,
    #     0,
    #     2150,
    #     2500,
    #     0.01,
    # )

    # _, emitplume_cube = simulate_satellite_images_with_plume(
    #     "AHSI",
    #     (100, 100),
    #     emit_plume,
    #     25,
    #     0,
    #     2150,
    #     2500,
    #     0.01,
    # )
    # np.save(
    #     "data/simulated_images/simulated_gaussianplume_1000_6_D.npy",
    #     gaussian_plume_cube,
    # )
    # np.save("data/simulated_images/simulated_emitplume.npy", emitplume_cube)

    import matplotlib.pyplot as plt

    def plot_gaussian_plume(
        file_path,
        x,
        y,
        vmin=None,
        vmax=None,
        title="Gaussian Plume",
        label="Concentration (ppm m)",
    ):
        concentration = np.load(file_path)
        X, Y = np.meshgrid(x, y)
        plt.figure(figsize=(10, 8))
        plt.contourf(
            X, Y, concentration.T, levels=50, cmap="Greys", vmin=vmin, vmax=vmax
        )
        plt.colorbar(label=label)
        plt.xlabel("pixel number")
        plt.ylabel("pixel number")
        plt.title(title)
        plt.show()

    plot_gaussian_plume(
        "data/simulated_images/emitplume1_2.npy",
        np.linspace(0, 100, 100),
        np.linspace(0, 100, 100),
        100,
        8000,
        title="EMIT Plume 1",
    )

    plot_gaussian_plume(
        "data/simulated_images/emitplume2_2.npy",
        np.linspace(0, 100, 100),
        np.linspace(0, 100, 100),
        100,
        8000,
        title="EMIT Plume 2",
    )

    plot_gaussian_plume(
        "data/simulated_images/gaussianplume_1000_10_D.npy",
        np.linspace(0, 100, 100),
        np.linspace(0, 100, 100),
        100,
        8000,
        title="Gaussian Plume",
    )

    plot_gaussian_plume(
        "data/simulated_images/GF5B_2300nm.npy",
        np.linspace(0, 100, 100),
        np.linspace(0, 100, 100),
        title="Radiance at 2300 nm",
        label="Radiance (uW/m^2/sr/nm)",
    )
