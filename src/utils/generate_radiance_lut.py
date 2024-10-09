import numpy as np
from scipy.interpolate import griddata

import sys
import os

sys.path.append("C://Users//RS//VSCode//matchedfiltermethod")
from utils.satellites_data.general_functions import get_simulated_satellite_radiance

# built a lookup table for radiance spectrum at different circumstances
# including different methane enhancement, different sensor height,
# different sensor viewing angle, different sensor azimuth angle, different solar zenith angle,
# different surface altitude,

# after built the lookup table, we can use the lookup table to get the radiance spectrum
# and then use the radiance spectrum to generate unit absorption spectrum at that
# circumstance, and then use the unit absorption spectrum to proceed the matched filter method


def generate_radiance_lut_for_satellite(satellitename: str):
    # Define parameter ranges
    methane_range = np.arange(0, 50000, 500)  # Methane concentration enhancement (ppm)
    altitude_range = np.arange(100, 900, 100)  # Satellite altitude (km)
    sza_range = np.arange(50, 90, 5)  # Solar zenith angle (degrees)

    # Initialize an empty dictionary to store the radiance spectra
    radiance_lookup_table = {}

    # Function to simulate radiance spectrum using MODTRAN (this is just a placeholder for actual MODTRAN calls)
    channels_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\satellite_channels\\{satellitename}_channels.npz"
    if not os.path.exists(f"{satellitename}_radiance_lookup_table.npz"):
        print("The satellite name is wrong")
        return None
    output_file = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\lookuptables\\{satellitename}_radiance_lookup_table.npz"

    def get_simulated_radiance(methane, altitude, sza):
        filename = f"{methane}_{altitude}_{sza}_tape7.txt"
        bands, radiance_spectrum = get_simulated_satellite_radiance(
            filename, channels_path, 900, 3000
        )
        return bands, radiance_spectrum

    def buildup_lut(output_file):
        for methane in methane_range:
            for altitude in altitude_range:
                for sza in sza_range:
                    # Simulate radiance spectrum
                    bands, simulated_radiance = get_simulated_radiance(
                        methane, altitude, sza
                    )

                    # Store the result in the lookup table with a tuple of parameters as the key
                    radiance_lookup_table[(methane, altitude, sza)] = simulated_radiance

        np.savez(
            output_file,
            wavelengths=bands,
            parameters=list(radiance_lookup_table.keys()),
            spectra=list(radiance_lookup_table.values()),
        )

        return radiance_lookup_table

    radiance_lookup_table = buildup_lut(output_file)

    return None


def load_radiance_lookup_table(filename: str):
    """
    Load the lookup table from a file.

    :param filename: Path to the file from which the lookup table will be loaded
    :return: Tuple of wavelengths and the lookup table (dictionary of enhancements and spectra)
    """
    data = np.load(filename)
    wavelengths = data["wavelengths"]
    parameters = data["parameters"]
    spectra = data["spectra"]
    lookup_table = {
        parameter: spectrum for parameter, spectrum in zip(parameters, spectra)
    }
    return wavelengths, lookup_table


# Interpolation function to estimate radiance for non-exact parameter values
def get_radiance_spectrum_from_lut(
    methane: float, altitude: float, sza: float, lookup_table: dict
):
    # Extract parameter points and corresponding radiance spectra
    points = np.array(list(lookup_table.keys()))
    values = np.array(list(lookup_table.values()))

    # Query point
    query_point = np.array([methane, altitude, sza])

    # Perform interpolation
    interpolated_radiance = griddata(points, values, query_point, method="linear")
    return interpolated_radiance
