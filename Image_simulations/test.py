import sys 
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from Tools import needed_function as nf
from MatchedFilter import matched_filter as mf
import numpy as np
from matplotlib import pyplot as plt


enhancements = np.arange(0,20500,500)
col_num = 100
row_num = 100
def matched_filter(base_array,data_array: np.array, unit_absorption_spectrum: np.array) :
    bands,rows,cols = data_array.shape
    # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
    background_spectrum = base_array
    target_spectrum = background_spectrum*unit_absorption_spectrum
    # 对当前目标光谱的每一行进行去均值操作，得到调整后的光谱，以此为基础计算协方差矩阵，并获得其逆矩阵
    radiancediff_with_back = data_array - background_spectrum[:,np.newaxis,np.newaxis]
    covariance = np.zeros((bands,bands))
    concentration = np.zeros((rows,cols))
    for i in range(rows):
        for j in range(cols):
            covariance += np.outer(radiancediff_with_back[:,i,j], radiancediff_with_back[:,i,j])
    covariance /= rows*cols
    covariance_inverse = np.linalg.inv(covariance)
    for i in range(rows):
        for j in range(cols):
            # 基于最优化公式计算每个像素的甲烷浓度增强值
            up = (radiancediff_with_back[:,i,j].T @ covariance_inverse @ target_spectrum)
            down = target_spectrum.T @ covariance_inverse @ target_spectrum
            concentration[i,j] = up / down
    return concentration

f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_0_ppmm_tape7.txt"
def process(filepath):
    channels_path=r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
    bands,radiance = nf.get_simulated_satellite_radiance(filepath,channels_path,900,2500)
    bands,_ = nf.get_simulated_satellite_radiance(filepath,channels_path,2150,2500)
    ahsi_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum.txt"
    uas = nf.open_unit_absorption_spectrum(ahsi_unit_absorption_spectrum_path)
    base_radiance, used_uas = nf.slice_data(radiance[:,np.newaxis,np.newaxis], uas, 2150, 2500)
    return base_radiance,used_uas,bands

for enhancement in enhancements:
    filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_{int(enhancement)}_ppmm_tape7.txt"
    used_radiance,used_uas,band = process(filepath)
    basefilepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_0_ppmm_tape7.txt"
    base_radiance,used_uas,band = process(filepath)
    simulated_noisyimage = np.zeros((len(band), row_num, col_num))
    for i in range(len(band)):  # 遍历每个波段
        current = used_radiance[i]
        noise = np.random.normal(0, current/100 , (row_num, col_num))  # 生成高斯噪声
        simulated_noisyimage[i,:,:] = np.ones((row_num,col_num))*current + noise  # 添加噪声到原始数据

    concentration = matched_filter(base_radiance[:,0,0],simulated_noisyimage,used_uas)
    hist_original, bin_edges_original = np.histogram(concentration.flatten(), bins=50)
    max_bin_index_original = np.argmax(hist_original)
    max_value_original = (bin_edges_original[max_bin_index_original] + bin_edges_original[max_bin_index_original + 1]) / 2
    print("enhancement is"+str(enhancement))
    print(max_value_original)
    # plt.subplot(1, 2, 1)
    # plt.title("Histogram of Original Image")
    # plt.hist(concentration.flatten(), bins=50, color='blue', alpha=0.7)
    # plt.axvline(max_value_original, color='red', linestyle='dashed', linewidth=1)
    # plt.text(max_value_original, max(hist_original), f'Mode: {max_value_original:.2f}', color='red', ha='right')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.show()
# filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_32000_ppmm_tape7.txt"
# channels_path=r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
# bands,radiance = nf.get_simulated_satellite_radiance(filepath,channels_path,900,2500)
# filepath2 = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_32000_ppmm_test_tape7.txt"
# channels_path=r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
# bands,radiance2 = nf.get_simulated_satellite_radiance(filepath2,channels_path,900,2500) 
# filepath3 = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_32000_ppmm_test1_tape7.txt"
# channels_path=r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
# bands,radiance3 = nf.get_simulated_satellite_radiance(filepath3,channels_path,900,2500) 
# plt.plot(bands,radiance,color='blue')
# plt.plot(bands,radiance2,color='red')
# plt.plot(bands,radiance3,color='green')
# plt.show()