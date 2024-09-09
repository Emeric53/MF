import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from MyFunctions.needed_function import get_simulated_satellite_radiance,read_simulated_radiance,load_satellite_channels,convolution,open_unit_absorption_spectrum
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def generate_uas_thenconvolve(satellite,enhancement_range,lower_wavelength,upper_wavelength):
    if satellite == "AHSI":
        channels_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
    elif satellite == "EMIT":
        channels_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\EMIT_channels.npz"
    else:
        print("Satellite name error!")
        return
    basepath = f"C:\\PcModWin5\\Bin\\batch\\{satellite}_Methane_0_ppmm_tape7.txt"
    bands,base_radiance = read_simulated_radiance(basepath)
    total_radiance = []
    for enhancement in enhancement_range:
        filepath = f"C:\\PcModWin5\\Bin\\batch\\{satellite}_Methane_{int(enhancement)}_ppmm_tape7.txt"
        _,radiance = read_simulated_radiance(filepath)
        current_convoluved_radiance = radiance/base_radiance
        total_radiance.append(current_convoluved_radiance)
    total_radiance = np.transpose(np.log(np.array(total_radiance)))
    slopelist = []
    for data in total_radiance:
        slope, intercept = np.polyfit(enhance_range, data, 1)
        slopelist.append(slope)
        # plt.scatter(enhance_range, data, label=f'Data {index + 1}')
        # plt.plot(enhance_range, slope * enhance_range + intercept, label=f'Fit {index + 1}', linestyle='--')
        # plt.legend()
        # plt.xlabel('Enhance Range')
        # plt.ylabel('Radiance')
        # plt.title('Original Data and Linear Fit')
        # plt.show()
    used_central_wavelengths, used_fwhms = load_satellite_channels(channels_path, lower_wavelength, upper_wavelength)
    convoluted_slope = convolution(used_central_wavelengths, used_fwhms, bands, slopelist)
    with open(f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum_alt.txt", 'w') as output:
        for index,data in enumerate(convoluted_slope):
            output.write(str(used_central_wavelengths[index])+' '+str(convoluted_slope[index])+'\n')
    return bands,slopelist


def generate_uas(satellite,enhancement_range,lower_wavelength,upper_wavelength):
    if satellite == "AHSI":
        channels_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_channels.npz"
    elif satellite == "EMIT":
        channels_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\EMIT_channels.npz"
    else:
        print("Satellite name error!")
        return
    
    total_radiance = []
    slopelist = []
    
    # filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_{enhance_range[0]}_ppmm_tape7.txt"
    # bands,_ = get_simulated_satellite_radiance(filepath,channels_path,lower_wavelength,upper_wavelength)
    
    for enhancement in enhancement_range:
        filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_{int(enhancement)}_ppmm_tape7.txt"
        bands,convoluved_radiance = get_simulated_satellite_radiance(filepath,channels_path,lower_wavelength,upper_wavelength)
        current_convoluved_radiance = [i for i in convoluved_radiance]
        current_convoluved_radiance = np.array(current_convoluved_radiance)
        total_radiance.append(current_convoluved_radiance)
    total_radiance = np.log(np.transpose(np.array(total_radiance)))
    
    # 拟合斜率作为单位吸收谱的结果
    for data in total_radiance:
        slope,_ = np.polyfit(enhance_range, data, 1)
        slopelist.append(slope)
    
    return bands,slopelist


def build_lookup_table():
    """build a lookup table for transmittance 

    Args:
        enhancements (np.array): the enhancement range of methane 

    Returns:
        np.array, dictionary: return the wavelengths and lookup_table 
    """
    lookup_table = {}
    enhancements = np.arange(-2000,50500,500)
    ahsi_channels_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_channels.npz"
    for enhancement in enhancements:
        filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_{int(enhancement)}_ppmm_tape7.txt"
        bands,convoluved_radiance = get_simulated_satellite_radiance(filepath,ahsi_channels_path,900,2500)
        current_convoluved_radiance = [i for i in convoluved_radiance]
        current_convoluved_radiance = np.array(current_convoluved_radiance)
        lookup_table[enhancement] = np.log(current_convoluved_radiance)
    return bands, lookup_table


def save_lookup_table(filename, wavelengths, lookup_table):
    """
    Save the lookup table to a file.

    :param filename: Path to the file where the lookup table will be saved
    :param wavelengths: List or array of wavelengths
    :param lookup_table: Dictionary where keys are enhancements and values are spectra
    """
    np.savez(filename, wavelengths=wavelengths, enhancements=list(lookup_table.keys()), spectra=list(lookup_table.values()))


def load_lookup_table(filename):
    """
    Load the lookup table from a file.

    :param filename: Path to the file from which the lookup table will be loaded
    :return: Tuple of wavelengths and the lookup table (dictionary of enhancements and spectra)
    """
    data = np.load(filename)
    wavelengths = data['wavelengths']
    enhancements = data['enhancements']
    spectra = data['spectra']
    lookup_table = {enhancement: spectrum for enhancement, spectrum in zip(enhancements, spectra)}
    return wavelengths, lookup_table


def load_needed_spectrum(startenhancement,endenhancement, wavelengths, lookup_table, low_wavelength, high_wavelength):
    """
    Interpolate the spectrum for a given enhancement within a specified wavelength range.

    :param enhancement: The enhancement value for which the spectrum is needed
    :param wavelengths: List or array of wavelengths
    :param lookup_table: Dictionary where keys are enhancements and values are spectra
    :param low_wavelength: Lower bound of the wavelength range
    :param high_wavelength: Upper bound of the wavelength range
    :return: Tuple of filtered wavelengths and interpolated spectrum
    """
    condition = (np.array(wavelengths) >= low_wavelength) & (np.array(wavelengths) <= high_wavelength)
    filtered_wavelengths = np.array(wavelengths)[condition]

    enhancements = np.array(list(lookup_table.keys()))
    enhance_range = (enhancements >= startenhancement) & (enhancements <= endenhancement)
    selected_enhancements = enhancements[enhance_range]

    # Filter and transpose the spectra based on selected enhancements and wavelength condition
    selected_spectra = np.array([lookup_table[enh] for enh in selected_enhancements])
    filtered_spectra = selected_spectra[:, condition]

    return filtered_wavelengths, np.transpose(filtered_spectra)


def generate_range_uas_AHSI(start_enhancement,end_enhancement,lower_wavelength,upper_wavelength):
    wavelengths, lookup_table = load_lookup_table("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\uas\\AHSI_lograd_lookup_table.npz")
    bands,total_radiance = load_needed_spectrum(start_enhancement,end_enhancement,wavelengths,lookup_table,lower_wavelength,upper_wavelength)
    slopelist = []

    # 拟合斜率作为单位吸收谱的结果
    enhancement_range = np.arange(start_enhancement,end_enhancement+500,500)
    for data in total_radiance:
        slope,_ = np.polyfit(enhancement_range, data, 1)
        slopelist.append(slope)
    
    return bands,slopelist


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
