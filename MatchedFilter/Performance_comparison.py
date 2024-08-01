import numpy as np
import sys 
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from Tools import needed_function as nf
from matplotlib import pyplot as plt


# def matched_filter(base_array,data_array: np.array, unit_absorption_spectrum: np.array) :
#     # 获取 以 波段 行数 列数 为顺序的数据
#     # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
#     background_spectrum = base_array
#     target_spectrum = background_spectrum*unit_absorption_spectrum
#     radiancediff_with_back = data_array - background_spectrum
#     covariance = np.outer(radiancediff_with_back, radiancediff_with_back)
#     covariance_inverse = np.linalg.inv(covariance)
#     # 基于最优化公式计算每个像素的甲烷浓度增强值
#     up = (radiancediff_with_back.T @ covariance_inverse @ target_spectrum)
#     down = target_spectrum.T @ covariance_inverse @ target_spectrum
#     concentration = up / down             
#     return concentration

def matched_filter(base_array,data_array: np.array, unit_absorption_spectrum: np.array,interval_unit_absorption_spectrum: np.array,interval_unit_absorption_spectrum2: np.array,interval_unit_absorption_spectrum3: np.array) :
    background_spectrum = base_array
    target_spectrum = background_spectrum*unit_absorption_spectrum
    concentration, _, _, _ = np.linalg.lstsq(target_spectrum[:, np.newaxis],(data_array - background_spectrum), rcond=None)    
    if concentration > 4600:
       background_spectrum = background_spectrum + 4600*target_spectrum
       target_spectrum = background_spectrum*interval_unit_absorption_spectrum
       concentration, _, _, _ = np.linalg.lstsq(target_spectrum[:, np.newaxis],(data_array - background_spectrum), rcond=None) 
       concentration += 5000
    if concentration > 9400:
        background_spectrum = background_spectrum + 4400*target_spectrum
        target_spectrum = background_spectrum*interval_unit_absorption_spectrum
        concentration, _, _, _ = np.linalg.lstsq(target_spectrum[:, np.newaxis],(data_array - background_spectrum), rcond=None) 
        concentration += 10000
    if concentration > 14200:
        background_spectrum = background_spectrum + 4200*target_spectrum
        target_spectrum = background_spectrum*interval_unit_absorption_spectrum
        concentration, _, _, _ = np.linalg.lstsq(target_spectrum[:, np.newaxis],(data_array - background_spectrum), rcond=None) 
        concentration += 15000
    return concentration


enhancements = np.arange(3500,5000,500)
enhancements = np.insert(enhancements,0,0)


# 读取单位吸收谱
ahsi_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum.txt"
uas = nf.open_unit_absorption_spectrum(ahsi_unit_absorption_spectrum_path)
ahsi_unit_absorption_spectrum_path2 = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum_2500to7500.txt"
uas2 = nf.open_unit_absorption_spectrum(ahsi_unit_absorption_spectrum_path2)
ahsi_interval_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\AHSI_unit_absorption_spectrum_from5000.txt"
interval_uas = nf.open_unit_absorption_spectrum(ahsi_interval_unit_absorption_spectrum_path)
ahsi_interval_unit_absorption_spectrum_path2 = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\AHSI_unit_absorption_spectrum_from10000.txt"
interval_uas2 = nf.open_unit_absorption_spectrum(ahsi_interval_unit_absorption_spectrum_path2)
ahsi_interval_unit_absorption_spectrum_path3 = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\AHSI_unit_absorption_spectrum_from15000.txt"
interval_uas3 = nf.open_unit_absorption_spectrum(ahsi_interval_unit_absorption_spectrum_path3)
base = None
concentration = 0
total_concentration = 0
biaslists = []
biaslists2 = []
enhancements = np.arange(0,20500,500)
for enhancement in enhancements:
    filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_{int(enhancement)}_ppmm_tape7.txt"
    channels_path=r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
    bands,radiance = nf.get_simulated_satellite_radiance(filepath,channels_path,900,2500)
    # 按照波长范围进行筛选，并获得slice 用于 radiance的筛选
    used_radiance, used_uas = nf.slice_data(radiance[:,np.newaxis,np.newaxis], uas, 2150, 2500)
    used_radiance, used_uas2 = nf.slice_data(radiance[:,np.newaxis,np.newaxis], uas2, 2150, 2500)
    _, used_interval_uas = nf.slice_data(radiance[:,np.newaxis,np.newaxis], interval_uas, 2150, 2500)
    _, used_interval_uas2 = nf.slice_data(radiance[:,np.newaxis,np.newaxis], interval_uas2, 2150, 2500)
    _, used_interval_uas3 = nf.slice_data(radiance[:,np.newaxis,np.newaxis], interval_uas3, 2150, 2500)
    # plt.plot(used_interval_uas)
    # plt.plot(used_interval_uas2)
    # plt.plot(used_interval_uas3)
    # plt.show()
    if base is None:
        base = used_radiance[:,0,0]
        continue
    else:
        concentration = matched_filter(base,used_radiance[:,0,0],used_uas,used_interval_uas,used_interval_uas2,used_interval_uas3)
        concentration2 = matched_filter(base,used_radiance[:,0,0],used_uas2,used_interval_uas,used_interval_uas2,used_interval_uas3)
        total_concentration += concentration
        print(concentration)
        print(concentration2)
        print(enhancement)
        biaslists.append(((concentration-enhancement)/enhancement)[0])
        biaslists2.append(((concentration2-enhancement)/enhancement)[0])
        print("bias is " + str((concentration-enhancement)/enhancement))
        print("bias is " + str((concentration2-enhancement)/enhancement))

plt.plot(enhancements[1:],biaslists)
plt.plot(enhancements[1:],biaslists2)
plt.savefig("bias.png")
print("total bias is "+ str(total_concentration/np.sum(enhancements)))







