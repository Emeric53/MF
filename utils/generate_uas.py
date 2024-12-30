import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

import os
# import sys


from utils.satellites_data.general_functions import (
    get_simulated_satellite_radiance,
)

# from utils.satellites_data.general_functions import open_unit_absorption_spectrum
# from utils.generate_radiance_lut_and_uas import (
#     generate_satellite_uas_for_specific_range_from_lut,
# )


# 基于modtran模拟结果，基于甲烷浓度增强范围，为特定卫星 生成单位吸收谱
def generate_satellite_uas(
    satellite: str,
    enhancement_range: np.ndarray,
    lower_wavelength: float,
    upper_wavelength: float,
):
    """
    generate the unit absorption spectrum for a given satellite

    Args:
        satellite (str): the satellite name
        enhancement_range (np.ndarray): the enhancement range of methane
        lower_wavelength (float): the lower bound of the wavelength range
        upper_wavelength (float): the upper bound of the wavelength range
    Returns:
        np.ndarray, np.ndarray: return the bands and slopelist
    """
    satellite_channels_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\satellite_channels\\{satellite}_channels.npz"
    if os.path.exists(satellite_channels_path):
        channels_path = satellite_channels_path
    else:
        print("The channels path does not exist.")

    total_radiance = []
    slopelist = []

    for enhancement in enhancement_range:
        # filepath = f"C:\\PcModWin5\\Bin\\batch\\{int(enhancement)}_90_0_tape7.txt"
        filepath = (
            f"C:\\PcModWin5\\Bin\\batch\\AHSI_methane_{int(enhancement)}_ppmm_tape7.txt"
        )
        bands, convoluved_radiance = get_simulated_satellite_radiance(
            filepath, channels_path, lower_wavelength, upper_wavelength
        )
        total_radiance.append(convoluved_radiance)
    total_radiance = np.transpose(np.log(np.array(total_radiance)))

    # 拟合斜率作为单位吸收谱的结果
    for data in total_radiance:
        slope, _ = np.polyfit(enhancement_range, data, 1)
        slopelist.append(min(slope, 0))

    return bands, slopelist


# 将单位吸收谱导出到文件
def export_uas_to_file(
    wavelengths: np.ndarray, slopelist: np.ndarray, output_file: str
):
    with open(output_file, "w") as output:
        for index, data in enumerate(slopelist):
            output.write(str(wavelengths[index]) + " " + str(data) + "\n")


# # 基于modtran模拟结果生成的查找表，基于甲烷浓度的头和尾，，为特定卫星生成单位吸收谱
# def generate_satellite_uas_for_specific_range_from_lut(
#     satellite_name: str,
#     start_enhancement: float,
#     end_enhancement: float,
#     lower_wavelength: float,
#     upper_wavelength: float,
#     sza: float,
#     altitude: float,
# ):
#     slopelist = []
#     total_radiance = []
#     wavelengths, _ = get_satellite_radiance_spectrum_from_lut(satellite_name, 0, 0, 0)
#     # 构建enhancement的范围
#     enhancement_range = generate_series_with_multiples_of_500(
#         start_enhancement, end_enhancement
#     )
#     condition = np.logical_and(
#         wavelengths >= lower_wavelength, wavelengths <= upper_wavelength
#     )
#     used_wavelengths = wavelengths[condition]
#     for enhancement in enhancement_range:
#         _, current_radiance = get_satellite_radiance_spectrum_from_lut(
#             satellite_name,
#             enhancement,
#             sza,
#             altitude,
#         )
#         total_radiance.append(np.log(current_radiance[condition]))
#     total_radiance = np.transpose(total_radiance)
#     for data in total_radiance:
#         slope, _ = np.polyfit(enhancement_range, data, 1)
#         slopelist.append(slope)

#     return used_wavelengths, slopelist


# def build_satellite_radiance_lut(enhancements: np.ndarray, satellite: str):
#     """build a lookup table for transmittance

#     Args:
#         enhancements (np.array): the enhancement range of methane

#     Returns:
#         np.array, dictionary: return the wavelengths and lookup_table
#     """
#     lookup_table = {}
#     channels_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\satellite_channels\\{satellite}_channels.npz"

#     for enhancement in enhancements:
#         filepath = (
#             f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_{int(enhancement)}_ppmm_tape7.txt"
#         )
#         bands, convoluved_radiance = get_simulated_satellite_radiance(
#             filepath, channels_path, 900, 2500
#         )

#         lookup_table[enhancement] = np.log(convoluved_radiance)
#     return bands, lookup_table


# def save_lookup_table(filename: str, wavelengths: np.ndarray, lookup_table: dict):
#     """
#     Save the lookup table to a file.

#     :param filename: Path to the file where the lookup table will be saved
#     :param wavelengths: List or array of wavelengths
#     :param lookup_table: Dictionary where keys are enhancements and values are spectra
#     """
#     np.savez(
#         filename,
#         wavelengths=wavelengths,
#         enhancements=list(lookup_table.keys()),
#         spectra=list(lookup_table.values()),
#     )


# def load_lookup_table(filename: str):
#     """
#     Load the lookup table from a file.

#     :param filename: Path to the file from which the lookup table will be loaded
#     :return: Tuple of wavelengths and the lookup table (dictionary of enhancements and spectra)
#     """
#     data = np.load(filename)
#     wavelengths = data["wavelengths"]
#     parameters = data["parameters"]
#     spectra = data["spectra"]
#     lookup_table = {
#         parameter: spectrum for parameter, spectrum in zip(parameters, spectra)
#     }
#     return wavelengths, lookup_table


# def load_needed_spectrum(
#     startenhancement,
#     endenhancement,
#     wavelengths,
#     lookup_table,
#     low_wavelength,
#     high_wavelength,
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
#     condition = (np.array(wavelengths) >= low_wavelength) & (
#         np.array(wavelengths) <= high_wavelength
#     )
#     filtered_wavelengths = np.array(wavelengths)[condition]

#     enhancements = np.array(list(lookup_table.keys()))
#     enhance_range = (enhancements >= startenhancement) & (
#         enhancements <= endenhancement
#     )
#     selected_enhancements = enhancements[enhance_range]

#     # Filter and transpose the spectra based on selected enhancements and wavelength condition

#     selected_spectra = np.array([lookup_table[enh] for enh in selected_enhancements])

#     filtered_spectra = selected_spectra[:, condition]

#     return filtered_wavelengths, np.transpose(filtered_spectra)


# def generate_satellite_uas_from_lut(
#     satellite_name: str,
#     start_enhancement: float,
#     end_enhancement: float,
#     lower_wavelength: float,
#     upper_wavelength: float,
# ):
#     wavelengths, lookup_table = load_lookup_table(
#         f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\uas\\{satellite_name}_lograd_lookup_table.npz"
#     )
#     bands, total_radiance = load_needed_spectrum(
#         start_enhancement,
#         end_enhancement,
#         wavelengths,
#         lookup_table,
#         lower_wavelength,
#         upper_wavelength,
#     )
#     slopelist = []

#     # 拟合斜率作为单位吸收谱的结果
#     enhancement_range = np.arange(start_enhancement, end_enhancement + 500, 500)
#     for data in total_radiance:
#         slope, _ = np.polyfit(enhancement_range, data, 1)
#         slopelist.append(slope)

#     return bands, slopelist


# for i in range(-2000,46000,2000):
#     enhance_range = np.arange(i,i+6500,500)
#     bands,slopelist= generate_uas("AHSI",enhance_range,900,2500)
#     # export the unit absorption spectrum result to a txt file
#     with open(f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\uas\\AHSI_UAS_{i}_{i+6000}.txt", 'w') as output:
#         for index,data in enumerate(slopelist):
#             output.write(str(bands[index])+' '+str(data)+'\n')

# 可视化
# fig, ax = plt.subplots(1,2)
# uas_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\uas\\AHSI_UAS_total.txt"
# bands,slopelist=open_unit_absorption_spectrum(uas_path,2100,2500)
# ax[0].plot(bands,slopelist)

# enhance_range = np.arange(2000,48000,2000)
# for enhance in enhance_range:
#     uas_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\uas\\AHSI_UAS_end_{enhance}.txt"
#     print(uas_path)
#     bands,slopelist=open_unit_absorption_spectrum(uas_path,2100,2500)
#     ax[1].plot(bands,slopelist)
# plt.show()

# fig, ax = plt.subplots(1,1)
# uas_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\uas\\AHSI_UAS_total.txt"
# bands,slopelist=open_unit_absorption_spectrum(uas_path,2100,2500)
# ax.plot(bands,50000*slopelist,color='blue')

# uas_path1 = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\uas\\AHSI_UAS_end_20000.txt"
# uas_path2 = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\uas\\AHSI_UAS_end_30000.txt"
# _,slopelist1=open_unit_absorption_spectrum(uas_path1,2100,2500)
# _,slopelist2=open_unit_absorption_spectrum(uas_path2,2100,2500)
# ax.plot(bands,20000*slopelist1*(((50000*slopelist2)/(20000*slopelist1))),color='red')
# plt.show()

if __name__ == "__main__":
    print("generate_uas")
    # wvls0, uas0 = open_unit_absorption_spectrum(
    #     r"C:\Users\RS\VSCode\matchedfiltermethod\src\data\uas_files\AHSI_UAS.txt",
    #     1500,
    #     2500,
    # )
    # wvls, uas = generate_satellite_uas("AHSI", np.arange(0, 50500, 500), 1500, 2500)
    # wvls1, uas1 = generate_satellite_uas_for_specific_range_from_lut(
    #     "AHSI", 0, 50000, 1500, 2500, 90, 0
    # )

    # ax, fig = plt.subplots(1, 3)
    # fig[0].plot(wvls, uas)
    # fig[1].plot(wvls1, uas1)
    # fig[2].plot(wvls0, uas0)
    # plt.savefig("uas_compare.png")

    def read_simulated_radiance(path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Read the simulated radiance data from the modtran output file and return the wavelengths and radiance.

        :param path: Path to the modtran output file
        :return: Tuple of wavelengths and radiance
        """
        with open(path, "r") as f:
            radiance_wvl = []
            radiance_list = []
            # read the lines containing the radiance data
            datalines = f.readlines()[11:-2]

            # wavenumber = np.array(datalines[:, :9], dtype=float)
            # radiance = np.array(datalines[:, 97:108], dtype=float)
            # wavelength = 10000000 / wavenumber
            # radiance = radiance * 10e7 / wavelength**2 * 10000

            for data in datalines:
                # convert the wavenumber to wavelength
                wvl = 10000000 / float(data[0:9])
                radiance_wvl.append(wvl)
                # convert the radiance in W/cm^2/sr/cm^-1 to W/m^2/sr/nm
                radiance = float(data[97:108]) * 10e7 / wvl**2 * 10000
                radiance_list.append(radiance)
        # reverse the order of the lists since the original order is according to the wavenumber
        # simulated_rad_wavelengths = wavelength[::-1]
        # simulated_radiance = radiance[::-1]
        simulated_rad_wavelengths = np.array(radiance_wvl)[::-1]
        simulated_radiance = np.array(radiance_list)[::-1]
        return simulated_rad_wavelengths, simulated_radiance

    channels_path = "data/satellite_channels/AHSI_channels.npz"
    filepath1 = "C:\\PcModWin5\\Bin\\batch\\25000_25_0_tape7.txt"
    # filepath = "C:\\PcModWin5\\Bin\\batch\\0_0_0_tape7.txt"
    filepath = "C:\\PcModWin5\\Bin\\batch\\AHSI_methane_20000_ppmm_tape7.txt"
    fig, ax = plt.subplots(1, 2)
    bands, convoluved_radiance = get_simulated_satellite_radiance(
        filepath, channels_path, 2100, 2500
    )
    ax[0].plot(bands, convoluved_radiance)
    bands, convoluved_radiance = get_simulated_satellite_radiance(
        filepath1, channels_path, 2100, 2500
    )
    ax[1].plot(bands, convoluved_radiance)

    # bands, radiance = read_simulated_radiance(filepath1)
    # condition = np.logical_and(bands >= 2100, bands <= 2500)
    # ax[0].plot(bands[condition], radiance[condition])
    # print(bands[condition].shape)
    # bands, radiance = read_simulated_radiance(filepath)
    # condition = np.logical_and(bands >= 2100, bands <= 2500)
    # ax[1].plot(bands[condition], radiance[condition])
    # print(bands[condition].shape)
    plt.show()
