import numpy as np

import sys

sys.path.append("C://Users//RS//VSCode//matchedfiltermethod//src")
from utils.satellites_data.general_functions import (
    open_unit_absorption_spectrum,
    get_simulated_satellite_radiance,
)
from utils.lookup_table import load_lookup_table, lookup_spectrum


# 基于查找表和浓度值获得透射率cube
def generate_transmittance_cube(
    plumes: np.ndarray, low_wavelength: float, high_wavelength: float
) -> np.ndarray:
    """
    Generate a transmittance cube based on the lookup table and concentration values.

    :param plumes: 2D NumPy array of concentration values
    :param low_wavelength: Lower bound of the wavelength range
    :param high_wavelength: Upper bound of the wavelength range
    :return: 3D NumPy array of transmittance values
    """
    loaded_wavelengths, loaded_lookup_table = load_lookup_table(
        "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_trans_lookup_table.npz"
    )
    used_wavelengths, _ = lookup_spectrum(
        0, loaded_wavelengths, loaded_lookup_table, low_wavelength, high_wavelength
    )
    transmittance_cube = np.ones(
        (len(used_wavelengths), plumes.shape[0], plumes.shape[1])
    )
    for i in range(plumes.shape[1]):
        for j in range(plumes.shape[0]):
            current_concentration = plumes[i, j]
            _, transmittance_cube[:, i, j] = lookup_spectrum(
                current_concentration,
                loaded_wavelengths,
                loaded_lookup_table,
                low_wavelength,
                high_wavelength,
            )
    return transmittance_cube


# 基于单位吸收谱和浓度值获得透射率cube
def generate_transmittance_cube_fromuas(
    plumes: np.ndarray, uas_path, low_wavelength, high_wavelength
):
    """
    Generate a transmittance cube based on unit absorption spectrum and concentration values.

    :param plumes: 2D NumPy array of concentration values
    :param low_wavelength: Lower bound of the wavelength range
    :param high_wavelength: Upper bound of the wavelength range
    :return: 3D NumPy array of transmittance values
    """
    _, uas = open_unit_absorption_spectrum(uas_path, low_wavelength, high_wavelength)
    transmittance_cube = np.ones((len(uas), plumes.shape[0], plumes.shape[1]))
    for i in range(plumes.shape[1]):
        for j in range(plumes.shape[0]):
            current_concentration = plumes[i, j]
            transmittance_cube[:, i, j] = 1 + uas * current_concentration
    return np.clip(transmittance_cube, 0, 1)


# 模拟卫星遥感影像
def simulate_satellite_images(
    radiance_path: str,
    satellite_name: str,
    plume: np.ndarray,
    lower_wavelength: float = 2150,
    upper_wavelength: float = 2500,
    noise_level=0.005,
):
    return None


# 模拟带甲烷烟羽的卫星遥感影像
def simulate_satellite_images_with_plumes(
    radiance_path: str,
    satellite_name: str,
    plume: np.ndarray,
    lower_wavelength: float = 2150,
    upper_wavelength: float = 2500,
    noise_level=0.005,
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
    if satellite_name == "emit":
        channels_path = (
            "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\EMIT_channels.npz"
        )
    elif satellite_name == "ahsi":
        channels_path = (
            "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_channels.npz"
        )
    else:
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
