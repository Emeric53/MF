import numpy as np
from scipy.integrate import trapz
# Description: This script demonstrates how to convolve a spectrum with a Gaussian response function.

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


# with open(r"C:\Users\RS\Desktop\All\EMIT_real.txt",'r') as rl:
#     datalines = rl.readlines()
#     for data in datalines:
#         real_radiance.append(float(data))

# 函数：高斯响应函数
def gaussian_response(wavelengths, center, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)


# 存储每个波段的卷积结果
band_radiance = []

# 计算每个波段的卷积积分
for center, fwhm in zip(center_wavelengths, fwhms):
    response = gaussian_response(simulated_wavelengths, center, fwhm)
    product = simulated_radiance * response
    integrated_radiance = trapz(product, simulated_wavelengths)
    band_radiance.append(integrated_radiance * 1000000)
band_radiance = np.array(band_radiance)
np.save("EMIT_band_radiance.npy", band_radiance)

