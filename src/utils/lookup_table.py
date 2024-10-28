import numpy as np
from scipy.interpolate import interp1d


# 从文件加载查找表
def load_lut(filename):
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
def lookup_from_lut(
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
    loaded_wavelengths, loaded_lookup_table = load_lut(
        "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_trans_lookup_table.npz"
    )
    used_wavelengths, _ = lookup_from_lut(
        0, loaded_wavelengths, loaded_lookup_table, low_wavelength, high_wavelength
    )
    transmittance_cube = np.ones(
        (len(used_wavelengths), plumes.shape[0], plumes.shape[1])
    )
    for i in range(plumes.shape[1]):
        for j in range(plumes.shape[0]):
            current_concentration = plumes[i, j]
            _, transmittance_cube[:, i, j] = lookup_from_lut(
                current_concentration,
                loaded_wavelengths,
                loaded_lookup_table,
                low_wavelength,
                high_wavelength,
            )
    return transmittance_cube


# 基于单位吸收谱和浓度值获得透射率cube
def generate_transmittance_cube_from_uas(
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
def simulate_images(
    spectrum: np.ndarray,
    noise_level=0.005,
    row_num=100,
    col_num=100,
):
    satellite_cube = spectrum[:, None, None] * np.ones(
        (spectrum.shape[0], row_num, col_num)
    )
    noise = np.zeros((spectrum.shape[0], row_num, col_num))
    for i in range(spectrum.shape[0]):
        noise[i, :, :] = np.random.normal(
            0, spectrum[i] * noise_level, (row_num, col_num)
        )
    noisy_satellite_cube = satellite_cube + noise
    return noisy_satellite_cube


# # ? 以前的旧代码
# # 从文件加载查找表
# def load_lut(filename):
#     """
#     Load the lookup table from a file.

#     :param filename: Path to the file from which the lookup table will be loaded
#     :return: Tuple of wavelengths and the lookup table (dictionary of enhancements and spectra)
#     """
#     data = np.load(filename)
#     wavelengths = data["wavelengths"]
#     enhancements = data["enhancements"]
#     spectrum = data["spectrum"]
#     lookup_table = {
#         enhancement: spectrum for enhancement, spectrum in zip(enhancements, spectrum)
#     }
#     return wavelengths, lookup_table


# # ? 以前的旧代码
# # 插值查找
# def lookup_from_lut(
#     enhancement, wavelengths, lookup_table, low_wavelength, high_wavelength
# ):
#     """
#     Interpolate the spectrum for a given enhancement within a specified wavelength range.

#     :param enhancement: The enhancement value for which the spectrum is needed
#     :param wavelengths: List or array of wavelengths
#     :param lookup_table: Dictionary where keys are enhancements and values are spectra
#     :param low_wavelength: Lower bound of the wavelength range
#     :param high_wavelength: Upper bound of the wavelength range
#     :return: Tuple of filtered wavelengths and interpolated spectrum
#     """
#     condition = np.where(
#         (np.array(wavelengths) >= low_wavelength)
#         & (np.array(wavelengths) <= high_wavelength)
#     )
#     enhancements = np.array(list(lookup_table.keys()))
#     spectrum = np.array(list(lookup_table.values()))[:, condition]
#     interpolator = interp1d(enhancements, spectrum, axis=0, fill_value="extrapolate")
#     result = np.clip(interpolator(enhancement)[0, :], 0, 1)
#     return np.array(wavelengths)[condition], result
