from scipy.signal import convolve
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

with open("C:\\Users\\RS\\Downloads\\SpectrMol_91609d00d88b68e018ee275935ba6dafaa46c084.xs.txt","r") as cs:
    data = cs.readlines()[6:]
    wavelengthlist = []
    crosssectionlist = []
    e_crosssectionlist = []
    for i in data:
        wavelength = 1/float(i[1:12])*10000
        wavelengthlist.append(wavelength)
        crosssection = float(i[16:32])
        e_crosssection = math.exp(-1*crosssection*math.pow(10,20))
        crosssectionlist.append(crosssection*math.pow(10,20))
        e_crosssectionlist.append(e_crosssection)
print(crosssectionlist)
with open("C:\\Users\\RS\\Downloads\\SpectrMol_cf49cccd8b7d702caf40fb3d979798e7ec87336a.rad.txt","r") as  rad:
    data = rad.readlines()[6:]
    wavelengthlist = []
    radlist = []
    for i in data:
        wavelength = 1 / float(i[1:12]) * 10000000
        wavelengthlist.append(wavelength)
        rad = float(i[16:32])
        radlist.append(rad)

with open("C:\\Users\\RS\\Desktop\\simu1.txt","r") as channels:
    data = channels.readlines()[8:]
    bandlist = []
    for i in data:
        band = float(i[2:12])
        bandlist.append(band)

csarray = np.array(e_crosssectionlist)
radarray = np.array(radlist)
wlarray = np.array(wavelengthlist)


sigma_conv = 8.546/2  # 分辨率为20nm，即标准差为10nm

convolved_channel = []
band_crosssection = []
for i in bandlist:
    convolving_term = stats.norm(i, sigma_conv)
    convolved_radiance = radarray*convolving_term.pdf(wavelengthlist)
    convolved_radiance = np.trapz(convolved_radiance,wavelengthlist)
    convolved_radiance_final = radarray*convolving_term.pdf(wavelengthlist)*csarray
    convolved_radiance_final = np.trapz(convolved_radiance_final,wavelengthlist)
    convolved_channel.append(convolved_radiance)
    band_crosssection.append(-1*math.log(convolved_radiance_final/convolved_radiance, math.e)/math.pow(10,20))

print(convolved_channel)
# 对光谱进行卷积
plt.plot(bandlist,band_crosssection)
plt.show()

# 创建一个新数组
new_array = []

# 对于原始数组中的每个位置
for i in range(len(band_crosssection)):
    # 计算与前一个元素的差值
    if i == 0:
        diff1 = abs(band_crosssection[i] - band_crosssection[i])
    else:
        diff1 = abs(band_crosssection[i] - band_crosssection[i - 1])

    # 计算与后一个元素的差值
    if i == len(band_crosssection) - 1:
        diff2 = abs(band_crosssection[i] - band_crosssection[i])
    else:
        diff2 = abs(band_crosssection[i] - band_crosssection[i + 1])

    # 取较大的差值作为新数组中的元素
    max_diff = max(diff1, diff2)

    # 将max_diff添加到新数组中
    new_array.append(max_diff)
contrast = np.array(new_array)
index = np.argmax(contrast)
print(index)
print(bandlist[index])
print(band_crosssection[index-1:index+2])
plt.plot(bandlist, contrast)
plt.xlabel="1"
plt.ylabel="2"
plt.show()
print(band_crosssection[index]-band_crosssection[index+1])