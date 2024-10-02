import numpy as np
import sys

sys.path.append("C://Users//RS//VSCode//matchedfiltermethod")
import needed_functions as nf
from scipy.interpolate import griddata

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
    if satellitename == "AHSI":
        channels_path = (
            "C://Users//RS//VSCode//matchedfiltermethod//MyData//AHSI_channels.npz"
        )
    elif satellitename == "EMIT":
        channels_path = (
            "C://Users//RS//VSCode//matchedfiltermethod//MyData//EMIT_channels.npz"
        )
    else:
        print("satellite name is not recognized")
        return
    output_file = f"{satellitename}_radiance_lookup_table.npz"

    def get_simulated_radiance(methane, altitude, sza):
        filename = f"{methane}_{altitude}_{sza}.flt"
        bands, radiance_spectrum = nf.get_simulated_satellite_radiance(
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


def load_lookup_table(filename: str):
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


def generate_series_with_multiples_of_500(start_value: float, end_value: float):
    """
    Generate a series starting with start_value, ending with end_value, and inserting multiples of 500 between them.

    :param start_value: The starting value of the series
    :param end_value: The ending value of the series
    :return: A list containing the series with multiples of 500 inserted between start_value and end_value
    """
    # Ensure start_value is less than or equal to end_value
    if start_value > end_value:
        start_value, end_value = end_value, start_value

    # Get the first multiple of 500 greater than or equal to start_value
    first_500_multiple = np.ceil(start_value / 500) * 500
    # Get the last multiple of 500 less than or equal to end_value
    last_500_multiple = np.floor(end_value / 500) * 500

    # Generate the series of 500 multiples between first_500_multiple and last_500_multiple
    multiples_of_500 = np.arange(first_500_multiple, last_500_multiple + 1, 500)

    # Combine start_value, multiples_of_500, and end_value, avoiding duplicates
    series = (
        [start_value] + multiples_of_500.tolist() + [end_value]
        if start_value != end_value
        else [start_value]
    )

    # Remove duplicate multiples of 500 that may have been added from start_value or end_value
    series = list(dict.fromkeys(series))

    return series


def generate_range_uas_for_specific_satellite_lut(
    satellite_name: str,
    start_enhancement: float,
    end_enhancement: float,
    lower_wavelength: float,
    upper_wavelength: float,
    altitude: float,
    sza: float,
):
    wavelengths, lookup_table = load_lookup_table(
        f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\uas\\{satellite_name}_radiance_lookup_table.npz"
    )

    slopelist = []
    total_radiance = []

    # 构建enhancement的范围
    enhancement_range = np.array(
        generate_series_with_multiples_of_500(start_enhancement, end_enhancement)
    )
    condition = np.logical_and(
        wavelengths >= lower_wavelength, wavelengths <= upper_wavelength
    )
    used_wavelengths = wavelengths[condition]
    for enhancement in enhancement_range:
        current_radiance = get_radiance_spectrum_from_lut(
            enhancement, altitude, sza, lookup_table
        )
        total_radiance.append(np.log(current_radiance[condition]))
    total_radiance = np.transpose(total_radiance)
    for data in total_radiance:
        slope, _ = np.polyfit(enhancement_range, data, 1)
        slopelist.append(slope)

    return used_wavelengths, slopelist


def export_uas_to_file(
    wavelengths: np.ndarray, slopelist: np.ndarray, output_file: str
):
    with open(output_file, "w") as output:
        for index, data in enumerate(slopelist):
            output.write(str(wavelengths[index]) + " " + str(data) + "\n")
