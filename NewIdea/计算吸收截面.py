import math
import matplotlib.pyplot as plt
import numpy
import numpy as np

with open("C:\\Users\\RS\\Desktop\\simu1.txt","r") as simu1:
    simu1data = simu1.readlines()[8:]
    wavelengthlist = []
    radiancelist1 = []
    for i in simu1data:
        wavelength = float(i[2:12])
        wavelengthlist.append(wavelength)
        radiance_1 = i[20:32]
        radiance_1 = float(radiance_1)*10000
        radiancelist1.append(radiance_1)
with open("C:\\Users\\RS\\Desktop\\simu2.txt", "r") as simu2:
    simu2data = simu2.readlines()[8:]
    wavelengthlist = []
    radiancelist2 = []
    for i in simu2data:
        wavelength = float(i[2:12])
        wavelengthlist.append(wavelength)
        radiance_2 = i[20:32]
        radiance_2 = float(radiance_2)*10000
        radiancelist2.append(radiance_2)
crosssectionlist = []
concentration = 1361854.5654121852
for one,two in zip(radiancelist1,radiancelist2):
    crosssectionlist.append(math.log(one/two, math.e)/10000000/concentration)
print(crosssectionlist)
# 创建一个新数组
new_array = []

# 对于原始数组中的每个位置
for i in range(len(crosssectionlist)):
    # 计算与前一个元素的差值
    if i == 0:
        diff1 = abs(crosssectionlist[i] - crosssectionlist[i])
    else:
        diff1 = abs(crosssectionlist[i] - crosssectionlist[i - 1])

    # 计算与后一个元素的差值
    if i == len(crosssectionlist) - 1:
        diff2 = abs(crosssectionlist[i] - crosssectionlist[i])
    else:
        diff2 = abs(crosssectionlist[i] - crosssectionlist[i + 1])

    # 取较大的差值作为新数组中的元素
    max_diff = max(diff1, diff2)

    # 将max_diff添加到新数组中
    new_array.append(max_diff)
contrast = np.array(new_array)
index = np.argmax(contrast)

plt.plot(wavelengthlist, crosssectionlist)
plt.xlabel="1"
plt.ylabel="2"
plt.show()

