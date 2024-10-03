import numpy as np
from scipy.interpolate import interp1d


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
    result = np.clip(interpolator(enhancement)[0, :], 0, 1)
    return np.array(wavelengths)[condition], result
