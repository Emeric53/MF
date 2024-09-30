import numpy as np
import sys

sys.path.append(r"C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from MyFunctions import needed_function as nf
from scipy.interpolate import interp1d

basefilepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_0_ppmm_tape7.txt"


channels_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_channels.npz"
_, base_radiance = nf.get_simulated_satellite_radiance(
    basefilepath, channels_path, 2100, 2500
)


def build_lookup_table(enhancements):
    """
    build a lookup table for transmittance

    Args:
        enhancements (np.array): the enhancement range of methane

    Returns:
        np.array,dictionary:  return the wavelengths and lookup_table
    """
    lookup_table = {}
    enhancements = np.arange(0, 50500, 500)
    ahsi_channels_path = (
        "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_channels.npz"
    )
    for enhancement in enhancements:
        filepath = (
            f"C:\\PcModWin5\\Bin\\batch\\AHSI_methane_{int(enhancement)}_ppmm_tape7.txt"
        )
        bands, spectrum = nf.get_simulated_satellite_radiance(
            filepath, ahsi_channels_path, 900, 2500
        )
        lookup_table[enhancement] = np.array(spectrum)
    return bands, lookup_table


# 保存查找表
def save_lookup_table(filename, wavelengths, lookup_table):
    """
    Save the lookup table to a file.

    :param filename: Path to the file where the lookup table will be saved
    :param wavelengths: List or array of wavelengths
    :param lookup_table: Dictionary where keys are enhancements and values are spectra
    """
    np.savez(
        filename,
        wavelengths=wavelengths,
        enhancements=list(lookup_table.keys()),
        spectra=list(lookup_table.values()),
    )


# 从文件加载查找表
def load_lookup_table(filename):
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
    result = interpolator(enhancement)[0, :]
    return np.array(wavelengths)[condition], result


# bands, lookup_table = build_lookup_table(np.arange(0,50500,500))
# save_lookup_table("C:\\Users\\RS\\VSCode\matchedfiltermethod\\MyData\\enhanced_radiance\\AHSI_rad_lookup_table.npz", bands, lookup_table)
# wvls,lut = load_lookup_table("C:\\Users\\RS\\VSCode\matchedfiltermethod\\MyData\\enhanced_radiance\\AHSI_rad_lookup_table.npz")
# wvl,result = lookup_spectrum(5000, wvls, lut, 900, 2500)
# from matplotlib import pyplot as plt
# plt.plot(wvl, result)
# plt.show()
