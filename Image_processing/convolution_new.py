import numpy as np
from scipy.integrate import trapz
from matplotlib import pyplot as plt

# 读取光谱数据 包括 波长 和被卷积的目标 例如 radiance 或 reflectance 或 transmittance
wvl_list = []
radiance_list = []
real_radiance = []

# 读取 modtran 模拟的光谱数据
with open(r"C:\PcModWin5\Bin\tape7", 'r') as f:
    datalines = f.readlines()[11:-2]
    for data in datalines:
        wvl = 10000000 / float(data[0:9])
        wvl_list.append(wvl)
        radiance = float(data[97:108])
        radiance_list.append(radiance)
simulated_wavelengths = np.array(wvl_list)[::-1]
simulated_radiance = np.array(radiance_list)[::-1]

# 读取高斯响应函数的中心波长和半高宽
with open(r"C:\Users\RS\Desktop\All\EMIT_wavelengths.csv", 'r') as wvl:
    central_wvl = wvl.readline().rstrip('\n').split(',')
    central_wvl = np.array(central_wvl)
    center_wavelengths = central_wvl.astype(np.float32)

with open(r"C:\Users\RS\Desktop\All\EMIT_fwhm.csv", 'r') as fwhm:
    fwhm = fwhm.readline().rstrip('\n').split(',')
    fwhm = np.array(fwhm)
    fwhms = fwhm.astype(np.float32)

def gaussian_response(wavelengths, center, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    response = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)
    normalization_factor = sigma * np.sqrt(2 * np.pi)
    return response / normalization_factor

def convolve_spectrum(center_wavelengths, fwhm_array, high_res_wavelengths, high_res_radiance):
    low_res_radiance = np.zeros(len(center_wavelengths))
    
    for i, (center, fwhm) in enumerate(zip(center_wavelengths, fwhm_array)):
        response = gaussian_response(high_res_wavelengths, center, fwhm)
        low_res_radiance[i] = np.trapz(high_res_radiance * response, high_res_wavelengths)
    
    return low_res_radiance


# 进行卷积
low_res_radiance = convolve_spectrum(center_wavelengths, fwhms, simulated_wavelengths, simulated_radiance)

# 输出结果
print("低分辨率辐射光谱数据:", low_res_radiance)
plt.plot(center_wavelengths, low_res_radiance, 'r')
plt.show()
