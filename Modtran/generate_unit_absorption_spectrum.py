import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from Tools.needed_function import get_simulated_satellite_radiance,read_simulated_radiance,load_satellite_channels,convolution
from matplotlib import pyplot as plt


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
        channels_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
    elif satellite == "EMIT":
        channels_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\EMIT_channels.npz"
    else:
        print("Satellite name error!")
        return
    total_radiance = []
    basepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_0_ppmm_tape7.txt"
    bands,base_radiance = get_simulated_satellite_radiance(basepath,channels_path,lower_wavelength,upper_wavelength)
    for enhancement in enhancement_range:
        filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_{int(enhancement)}_ppmm_tape7.txt"
        _,convoluved_radiance = get_simulated_satellite_radiance(filepath,channels_path,lower_wavelength,upper_wavelength)
        current_convoluved_radiance = [i for i in convoluved_radiance]
        current_convoluved_radiance = np.array(current_convoluved_radiance)/base_radiance
        total_radiance.append(current_convoluved_radiance)
    total_radiance = np.log(np.transpose(np.array(total_radiance)))
    slopelist = []
    for index,data in enumerate(total_radiance):
        slope, intercept = np.polyfit(enhance_range, data, 1)
        slopelist.append(slope)
        # # 绘制原始数据点
        # plt.scatter(enhance_range, data, label=f'Data {index + 1}')     
        # # 绘制拟合直线
        # plt.plot(enhance_range, slope * enhance_range + intercept, label=f'Fit {index + 1}', linestyle='--')
        # plt.legend()
        # plt.xlabel('Enhance Range')
        # plt.ylabel('Radiance')
        # plt.title('Original Data and Linear Fit')
        # plt.show()
    # export the unit absorption spectrum result to a txt file
    with open(f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\{satellite}_unit_absorption_spectrum.txt", 'w') as output:
        for index,data in enumerate(slopelist):
            output.write(str(bands[index])+' '+str(data)+'\n')
    return bands,slopelist




# 读取
interval_range = np.array([5000,7500,10000])
enhance_range = np.arange(0,20500,500)
bands,slopelist= generate_uas("EMIT",enhance_range,900,2500)

# 创建主图
fig, ax1 = plt.subplots()

ax1.plot(bands,slopelist, 'r', label='slope')
ax1.set_xlabel('wvl')
ax1.set_ylabel('slope', color='k')
ax1.tick_params(axis='y', labelcolor='k')
ax1.legend(loc='upper left')
plt.show()

