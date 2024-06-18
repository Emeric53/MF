# Description: This script demonstrates how to convolve a spectrum with a Gaussian response function.
import numpy as np
from scipy.integrate import trapz
from matplotlib import pyplot as plt

# 函数：高斯响应函数
def gaussian_response(wavelengths, center, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    normalization_factor = sigma * np.sqrt(2 * np.pi)
    response = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)/normalization_factor
    return response

def convolution(center_wavelengths:list, fwhms:list, raw_wvls:list, raw_data:list):
        # 存储每个波段的卷积结果
    convoluved_data = []
    # 计算每个波段的卷积积分
    for center, fwhm in zip(center_wavelengths, fwhms):
        response = gaussian_response(raw_wvls, center, fwhm)
        product = raw_data * response
        integrated_data = trapz(product, raw_wvls)
        convoluved_data.append(integrated_data)
    return np.array(convoluved_data)

# 读取高斯响应函数的中心波长和半高宽
with open(r"C:\Users\RS\Desktop\All\EMIT_wavelengths.csv", 'r') as wvl:
    central_wvls = wvl.readline().rstrip('\n').split(',')
    central_wvls = np.array(central_wvls).astype(np.float32)

with open(r"C:\Users\RS\Desktop\All\EMIT_fwhm.csv", 'r') as fwhm:
    fwhms = fwhm.readline().rstrip('\n').split(',')
    fwhms = np.array(fwhms).astype(np.float32)

# select the central wavelengths and FWHMs that are within the range of 1000-2500 nm
index = np.where((central_wvls >= 1000) & (central_wvls <= 2500))
used_center_wavelengths = central_wvls[index]
used_fwhms = fwhms[index]

# radiance 卷积
# 读取 modtran 模拟的光谱数据
with open(r"C:\PcModWin5\Usr\Radiance_simulation_with_otherhyperspectral.fl7", 'r') as f:
    radiance_wvl = []
    radiance_list = []
    datalines = f.readlines()[11:-2]
    for data in datalines:
        wvl = 10000000 / float(data[0:9])
        radiance_wvl.append(wvl)
        radiance = float(data[97:108])*10e7/wvl**2*10000
        radiance_list.append(radiance)
simulated_rad_wavelengths = np.array(radiance_wvl)[::-1]
simulated_radiance = np.array(radiance_list)[::-1]


with open(r"C:\\PcModWin5\\Usr\\FullSpecTran.fl7", 'r') as f:
    trans_wvl = []
    trans_list = []
    datalines = f.readlines()[12:-2]
    for data in datalines:
        wvl = 10000000 / float(data[0:9])
        trans_wvl.append(wvl)
        transmittance = float(data[108:115])
        trans_list.append(transmittance)
simulated_trans_wavelengths = np.array(trans_wvl)[::-1]
simulated_trans = np.array(trans_list)[::-1]



# with open(r"C:\Users\RS\Desktop\All\EMIT_real.txt",'r') as rl:
#     datalines = rl.readlines()
#     for data in datalines:
#         real_radiance.append(float(data))


convoluved_radiance = convolution(used_center_wavelengths,used_fwhms,simulated_rad_wavelengths,simulated_radiance)
convoluved_trans = convolution(used_center_wavelengths,used_fwhms,simulated_trans_wavelengths,simulated_trans)

# 顶部大图
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
ln1 = ax1.plot(simulated_rad_wavelengths, simulated_radiance, label='Mod, FWHM ~ 0.2nm', color='gray', alpha=0.6)
ln2 = ax1.plot(used_center_wavelengths, convoluved_radiance, label='EMIT, FWHM ~ 8nm', color='blue', alpha=0.6)
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)')
ax1.set_ylim(0, 0.6)
ax1.set_xlim(1000, 2500)
ax1.grid(True)

ax1_left = ax1.twinx()
ln3 = ax1_left.plot(simulated_trans_wavelengths, simulated_trans, label='Methane transmittance, FWHM ~ 0.2nm', color='red', linestyle='--', alpha=0.6)
ln4 = ax1_left.plot(used_center_wavelengths, convoluved_trans, label='Methane transmittance EMIT, FWHM ~ 8nm', color='green', linestyle='--', alpha=0.6)
ax1_left.set_ylabel('Transmittance')
ax1_left.set_ylim(0.4, 1)  # 根据实际透射率数据范围调整
ax1_left.set_xlim(1000, 2500)

# 合并图例并放在图表外部
lns = ln1 + ln2 + ln3 + ln4
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

plt.title('Radiance and Transmission over Wavelength')
plt.tight_layout(rect=[0, 0.05, 1, 1])


# 左下角小图
ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=1)
ax2.plot(simulated_trans_wavelengths, simulated_trans, color='gray')
ax2.plot(used_center_wavelengths, convoluved_radiance, color='blue')
ax2.set_xlim([1600, 1900])
ax2.set_ylim([0, 0.15])
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Radiance (W m⁻² sr⁻¹ nm⁻¹)')
ax2.grid(True)

# 右下角小图
ax3 = plt.subplot2grid((3, 3), (2, 2), colspan=1)
ax3.plot(simulated_trans_wavelengths, simulated_trans, color='gray')
ax3.plot(used_center_wavelengths, convoluved_radiance, color='blue')
ax3.set_xlim([2100, 2500])
ax3.set_ylim([0, 0.025])
ax3.set_xlabel('Wavelength (nm)')
ax3.grid(True)


plt.show()
