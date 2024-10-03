import numpy as np
from scipy.interpolate import interp1d

import sys

sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src")
from utils.satellites_data.general_functions import open_unit_absorption_spectrum


# 从文件加载查找表
def load_lookup_table(filename: str):
    """
    Load the lookup table from a file.

    :param filename: Path to the file from which the lookup table will be loaded
    :return: Tuple of wavelengths and the lookup table (dictionary of enhancements and spectra)
    """
    data = np.load(filename)
    wavelengths = data["wavelengths"]
    enhancements = data["enhancements"]
    spectra = data["spectra"]
    lookup_table = {
        enhancement: spectrum for enhancement, spectrum in zip(enhancements, spectra)
    }
    return wavelengths, lookup_table


# 插值查找
def lookup_spectrum(
    enhancement, wavelengths, lookup_table, low_wavelength, high_wavelength
):
    """
    Interpolate the spectrum for a given enhancement within a specified wavelength range.

    :param enhancement: The enhancement value for which the spectrum is needed
    :param wavelengths: List or array of wavelengths
    :param lookup_table: Dictionary where keys are enhancements and values are spectra
    :param low_wavelength: Lower bound of the wavelength range
    :param high_wavelength: Upper bound of the wavelength range
    :return: Tuple of filtered wavelengths and interpolated spectrum
    """
    condition = np.where(
        (np.array(wavelengths) >= low_wavelength)
        & (np.array(wavelengths) <= high_wavelength)
    )
    enhancements = np.array(list(lookup_table.keys()))
    spectra = np.array(list(lookup_table.values()))[:, condition]
    interpolator = interp1d(enhancements, spectra, axis=0, fill_value="extrapolate")
    result = np.clip(interpolator(enhancement)[0, :], 0, 1)
    return np.array(wavelengths)[condition], result


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


# 从 NumPy 数组中筛选数据并获取切片
def filter_and_slice(array: np.ndarray, min_val: float, max_val: float):
    """
    根据最大最小值阈值，筛选数组并获取原数组的切片。

    :param arr: 输入的 NumPy 数组
    :param min_val: 最小值阈值
    :param max_val: 最大值阈值
    :return: 筛选后的数组和原数组中的切片
    """
    condition = (array >= min_val) & (array <= max_val)
    filtered_arr = array[condition]
    nonzero_indices = np.nonzero(condition)[0]
    if len(nonzero_indices) == 0:
        return filtered_arr, None
    slice_start = nonzero_indices[0]
    slice_end = nonzero_indices[-1] + 1
    arr_slice = slice(slice_start, slice_end)
    return filtered_arr, arr_slice


# 基于波长最低值和最高值对radiance和uas进行切片
def slice_data(radiance_array, uas, low, high):
    """
    Slice radiance and unit absorption spectrum data based on the lowest and highest wavelengths.

    :param radiance_array: NumPy array of radiance data
    :param uas: NumPy array of unit absorption spectrum data
    :param low: Lower bound of the wavelength range
    :param high: Upper bound of the wavelength range
    :return: Tuple of sliced radiance array and used unit absorption spectrum
    """
    _, slice_uas = filter_and_slice(uas[:, 0], low, high)
    used_radiance = radiance_array[slice_uas, :, :]
    used_uas = uas[slice_uas, 1]
    return used_radiance, used_uas


# 模拟带甲烷烟羽的卫星遥感影像
def satellite_images_with_plumes_simulation(
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


if __name__ == "__main__":
    ahsi_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_unit_absorption_spectrum.txt"
    # 读取单位吸收谱
    _, uas = open_unit_absorption_spectrum(
        ahsi_unit_absorption_spectrum_path, 2100, 2500
    )
    # path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_trans_lookup_table"
    # low_wavelength = 1500
    # high_wavelength = 2500
    # cube = generate_transmittance_cube_fromuas(np.array([[10000],]),900,2500)

    # filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_0_ppmm_tape7.txt"
    # channels_path=r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
    # bands,radiance = get_simulated_satellite_radiance(filepath,channels_path,900,2500)
    # filepath2 = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_10000_ppmm_tape7.txt"
    # channels_path=r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
    # bands,radiance2 = get_simulated_satellite_radiance(filepath2,channels_path,900,2500)

    # channels,orginal_radiance = read_simulated_radiance(filepath)
    # channels,orginal_radiance2 = read_simulated_radiance(filepath2)
    # # from matplotlib import pyplot as plt
    # # # plt.plot(channels,orginal_radiance)
    # # # plt.plot(channels,orginal_radiance2)
    # # # plt.plot(bands,radiance)
    # # # plt.plot(bands,radiance2)
    # # plt.plot(channels,orginal_radiance2/orginal_radiance)
    # # plt.plot(bands,radiance2/radiance)
    # # plt.ylim(0,1.01)
    # # plt.show()

    # from matplotlib import pyplot as plt
    # # 计算比值
    # ratio = radiance * cube[:, 0, 0] / radiance2

    # # 创建主图
    # fig, ax1 = plt.subplots()

    # # 绘制 radiance*cube[:,0,0] 和 radiance2
    # ax1.plot(bands,cube[:, 0, 0], 'r', label='Radiance * Cube')
    # ax1.plot(bands,radiance2/radiance, 'b', label='Radiance2')
    # ax1.set_xlabel('Index')
    # ax1.set_ylabel('Radiance', color='k')
    # ax1.tick_params(axis='y', labelcolor='k')
    # # 添加图例
    # ax1.legend(loc='upper left')

    # # 创建第二个 y 轴
    # ax2 = ax1.twinx()
    # ax2.plot(bands,ratio, 'g', label='Ratio')
    # ax2.set_ylabel('Ratio', color='g')
    # ax2.tick_params(axis='y', labelcolor='g')
    # ax2.set_ylim(0, 1)

    # # 添加图例
    # ax2.legend(loc='upper right')

    # plt.title('Radiance and Ratio Plot')
    # plt.show()
