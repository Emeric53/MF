import numpy as np


# generate a function to ulitize DOAS method to retrieve methane concentration
def DOAS_method(
    spectra: np.ndarray, wavelengths: np.ndarray, reference_spectrum: np.ndarray
) -> np.ndarray:
    """
    Retrieve the methane concentration using the Differential Optical Absorption Spectroscopy (DOAS) method.

    :param spectra: 2D NumPy array of spectra
    :param wavelengths: 1D NumPy array of wavelengths
    :param reference_spectrum: 1D NumPy array of the reference spectrum
    :return: 2D NumPy array of retrieved methane concentrations
    """
    # calculate the difference between the spectra and the reference spectrum
    difference = spectra - reference_spectrum

    # calculate the sum of the differences
    sum_difference = np.sum(difference, axis=1)

    # calculate the sum of the reference spectrum
    sum_reference = np.sum(reference_spectrum)

    # calculate the retrieved methane concentrations
    methane_concentration = sum_difference / sum_reference

    return methane_concentration
