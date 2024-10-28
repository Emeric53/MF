import numpy as np

import os
import sys

sys.path.append("C://Users//RS//VSCode//matchedfiltermethod//src")
from utils.satellites_data.general_functions import (
    get_simulated_satellite_radiance,
)
from utils.generate_radiance_lut_and_uas import (
    batch_get_radiance_from_lut,
)


# ! 模拟带甲烷烟羽的卫星遥感影像 大小和 烟羽数组大小一致
def simulate_satellite_images(
    radiance_path: str,
    satellite_name: str,
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
    channels_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\satellite_channels\\{satellite_name}_channels.npz"
    if not os.path.exists(channels_path):
        print("The satellite name is not supported.")
        return None

    bands, simulated_convolved_spectrum = get_simulated_satellite_radiance(
        radiance_path, channels_path, lower_wavelength, upper_wavelength
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


def simulate_satellite_images_with_plume(
    satellite_name: str,
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

    row, col = plume.shape
    plume_flat = plume.flatten()

    wvls, radiance_cube = batch_get_radiance_from_lut(
        satellite_name, plume_flat, sza, altitude
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
    return noisy_radiance_cube
