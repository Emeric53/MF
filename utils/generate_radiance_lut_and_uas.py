import numpy as np
from scipy.interpolate import griddata
import time

import sys
import os

sys.path.append("C://Users//RS//VSCode//matchedfiltermethod")
from utils.satellites_data import general_functions as gf
# built a lookup table for radiance spectrum at different circumstances
# including different methane enhancement, different sensor height,
# different sensor viewing angle, different sensor azimuth angle, different solar zenith angle,
# different surface altitude,

# after built the lookup table, we can use the lookup table to get the radiance spectrum
# and then use the radiance spectrum to generate unit absorption spectrum at that
# circumstance, and then use the unit absorption spectrum to proceed the matched filter method


def generate_radiance_lut_for_satellite(satellitename: str):
    # Define parameter ranges
    methane_range = np.arange(0, 50500, 500)  # Methane concentration enhancement (ppm)
    sza_range = np.arange(0, 95, 5)  # Solar zenith angle (degrees)
    surface_altitude_range = np.arange(0, 6, 1)  # Surface

    # Initialize an empty dictionary to store the radiance spectra

    # Function to simulate radiance spectrum using MODTRAN (this is just a placeholder for actual MODTRAN calls)
    channels_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\data\\satellite_channels\\{satellitename}_channels.npz"
    if not os.path.exists(channels_path):
        print("The satellite name is wrong")
        return None
    output_file = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\data\\lookuptables\\{satellitename}_radiance_lookup_table.npz"

    def get_simulated_radiance(methane, altitude, sza):
        filename = rf"C:\PcModWin5\Bin\batch_result\batch\{int(methane)}_{int(sza)}_{int(altitude)}_tape7.txt"
        bands, radiance_spectrum = gf.get_simulated_satellite_radiance(
            filename, channels_path, 1500, 2500
        )
        return bands, radiance_spectrum

    def buildup_lut(output_file):
        radiance_lookup_table = {}
        for methane in methane_range:
            for altitude in surface_altitude_range:
                for sza in sza_range:
                    # Simulate radiance spectrum
                    bands, simulated_radiance = get_simulated_radiance(
                        methane, altitude, sza
                    )

                    # Store the result in the lookup table with a tuple of parameters as the key
                    radiance_lookup_table[(methane, sza, altitude)] = simulated_radiance

        np.savez(
            output_file,
            wavelengths=bands,
            parameters=list(radiance_lookup_table.keys()),
            spectrum=list(radiance_lookup_table.values()),
        )

        return radiance_lookup_table

    radiance_lookup_table = buildup_lut(output_file)

    return radiance_lookup_table


# ! 基于 甲烷浓度 SZA 地面高程 查询辐射光谱
def load_satellite_radiance_lookup_table(satellitename: str):
    """
    Load the lookup table from a file.

    :param filename: Path to the file from which the lookup table will be loaded
    :return: Tuple of wavelengths and the lookup table (dictionary of enhancements and spectra)
    """
    filename = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\data\\lookuptables\\{satellitename}_radiance_lookup_table.npz"
    data = np.load(filename)
    wavelengths = data["wavelengths"]
    parameters = data["parameters"]
    spectrum = data["spectrum"]
    lookup_table = {
        tuple(parameter): spectrum for parameter, spectrum in zip(parameters, spectrum)
    }
    return wavelengths, lookup_table


# Interpolation function to estimate radiance for non-exact parameter values
def get_satellite_radiance_spectrum_from_lut(
    satellitename: str, methane: float, sza: float, altitude: float, low, high
):
    wvls, lookup_table = load_satellite_radiance_lookup_table(satellitename)
    # Extract parameter points and corresponding radiance spectra
    points = np.array(list(lookup_table.keys()))
    values = np.array(list(lookup_table.values()))

    # Query point
    query_point = np.array([methane, sza, altitude])

    # Perform interpolation
    interpolated_radiance = griddata(points, values, query_point, method="linear")[0, :]
    condition = np.logical_and(wvls >= low, wvls <= high)
    wvls = wvls[condition]
    interpolated_radiance = interpolated_radiance[condition]
    return wvls, interpolated_radiance


# 批量化查询
def batch_get_radiance_from_lut(
    satellite_name, enhancement_range, sza, altitude, low=1500, high=2500
):
    # 加载查找表
    wavelengths, lookup_table = load_satellite_radiance_lookup_table(satellite_name)

    # 构建查询点 (甲烷增强, sza, 高度)
    query_points = np.array([[enh, sza, altitude] for enh in enhancement_range])

    # 批量插值查询
    radiance_list = griddata(
        list(lookup_table.keys()),
        list(lookup_table.values()),
        query_points,
        method="linear",
    )
    condition = np.logical_and(wavelengths >= low, wavelengths <= high)
    wavelengths = wavelengths[condition]
    radiance_list = np.array(radiance_list)[:, condition]
    return wavelengths, radiance_list


# 基于头尾浓度增强值，生成甲烷浓度增强范围
def generate_series_with_multiples_of_500(
    start_value: float, end_value: float
) -> np.ndarray:
    """
    Generate a numpy array starting and ending with the closest multiples of 500
    to start_value and end_value, and with 500 as the step size.

    :param start_value: The starting value
    :param end_value: The ending value
    :return: A numpy array with the series of 500 multiples
    """
    # Ensure start_value is less than or equal to end_value
    if start_value > end_value:
        start_value, end_value = end_value, start_value

    # Find the closest multiple of 500 greater than or equal to start_value
    first_500_multiple = np.ceil(start_value / 500) * 500

    # Find the closest multiple of 500 less than or equal to end_value
    last_500_multiple = np.floor(end_value / 500) * 500

    # Generate the multiples of 500 between first_500_multiple and last_500_multiple
    multiples_of_500 = np.arange(first_500_multiple, last_500_multiple + 1, 500)

    return multiples_of_500


def generate_satellite_uas_for_specific_range_from_lut(
    satellite_name: str,
    start_enhancement: float,
    end_enhancement: float,
    lower_wavelength: float,
    upper_wavelength: float,
    sza: float,
    altitude: float,
):
    # 1. 构建enhancement的范围
    if start_enhancement < 0:
        start_enhancement = 0
    if end_enhancement > 50000:
        end_enhancement == 50000
    enhancement_range = generate_series_with_multiples_of_500(
        start_enhancement, end_enhancement
    )

    # 2. 查询所有甲烷增强值的光谱（批量处理）
    wavelengths, radiance_list = batch_get_radiance_from_lut(
        satellite_name, enhancement_range, sza, altitude
    )
    # radiance_list /= radiance_list[0, :]
    # 3. 过滤波长范围
    condition = np.logical_and(
        wavelengths >= lower_wavelength, wavelengths <= upper_wavelength
    )
    used_wavelengths = wavelengths[condition]
    total_radiance = np.log(radiance_list[:, condition])
    print(total_radiance.shape)
    total_radiance = total_radiance

    # 4. 使用多项式拟合计算斜率（矢量化处理）
    slopelist = np.polyfit(enhancement_range, total_radiance, deg=1)[0]

    # slopelist = []
    # model_no_intercept = LinearRegression(fit_intercept=False)
    # enhancement_range = (enhancement_range - np.min(enhancement_range)).reshape(-1, 1)
    # for i in range(total_radiance.shape[1]):
    #     total_radiance[:, i] = -(total_radiance[:, i] - np.min(total_radiance[:, i]))
    #     model_no_intercept.fit(enhancement_range, total_radiance[:, i])
    #     slopelist.append(model_no_intercept.coef_[0])
    return (used_wavelengths, slopelist)


if __name__ == "__main__":
    pass
    # print("Start generating radiance lookup table")
    # generate_radiance_lut_for_satellite("AHSI")
    # generate_radiance_lut_for_satellite("EnMAP")
    # generate_radiance_lut_for_satellite("EMIT")
    # generate_radiance_lut_for_satellite("PRISMA")
    # generate_radiance_lut_for_satellite("ZY1")
    start = time.time()
    wvls, slope = generate_satellite_uas_for_specific_range_from_lut(
        "AHSI", 0, 50000, 2150, 2500, 60, 0
    )
    end = time.time()
    print(f"Time cost: {end - start}")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(wvls, np.array(slope))
    plt.show()
    # wvls, slope = generate_satellite_uas_for_specific_range_from_lut(
    #     "AHSI",
    #     start_enhancement=0,
    #     end_enhancement=5000,
    #     lower_wavelength=2150,
    #     upper_wavelength=2500,
    #     sza=60,
    #     altitude=0,
    # )
    # wvls, slope1 = generate_satellite_uas_for_specific_range_from_lut(
    #     "AHSI",
    #     start_enhancement=0,
    #     end_enhancement=25000,
    #     lower_wavelength=2150,
    #     upper_wavelength=2500,
    #     sza=60,
    #     altitude=0,
    # )
    # wvls, slope2 = generate_satellite_uas_for_specific_range_from_lut(
    #     "AHSI",
    #     start_enhancement=0,
    #     end_enhancement=50000,
    #     lower_wavelength=2150,
    #     upper_wavelength=2500,
    #     sza=60,
    #     altitude=0,
    # )
