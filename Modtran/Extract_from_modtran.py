import math
import numpy as np
import matplotlib.pyplot as plt
total_radiance = []
original_radiance = []
wavelengthlist = []
# 打开原始文件进行读取
# with open("C:\\Users\\RS\\Desktop\\AHSI\\AHSI_0.chn", 'r', encoding='utf-8') as file:
#         lines = file.readlines()[5:]
# for i in lines:
#         radiance = i[60:72]
#         wavelength = i[2:12]
#         radiance = float(radiance) * 1000000/8.9767
#         original_radiance.append(radiance)
#         wavelengthlist.append(wavelength)
#
# plt.plot(wavelengthlist, original_radiance)
# plt.xlabel('Wavelength(nm)')
# plt.ylabel("Unit absorption spectrum(ppm*m-1)")
# plt.show()

# 读取100个文件
for j in range(0, 101):
    file_name = f"C:\\Users\\RS\\Desktop\\AHSI\\AHSI_{j*100}.chn"  # 生成文件名
    with open(file_name, "r") as file:
        rawdata = file.readlines()[5:]
    final = {}
    wavelengthlist = []
    radiancelist = []
    for i in rawdata:
        wavelength = float(i[2:12])
        wavelengthlist.append(wavelength)
        radiance = i[60:72]
        radiance = float(radiance)*10000/8.9767
        radiance_log = math.log(radiance, math.e)
        radiancelist.append(radiance_log)
    total_radiance.append(radiancelist)

total_radiance = np.array(total_radiance)
total_radiance = np.transpose(total_radiance)

x = []
for i in range(101):
    x.append(i*100)
x = np.array(x)
slopelist = []
for index in range(len(wavelengthlist)):
    data = total_radiance[index]
    # 使用polyfit函数进行线性回归拟合
    slope, intercept = np.polyfit(x, data, 1)
    slopelist.append(slope)
for i in slopelist:
    if i <= -0.01:
        print(i)

plt.figure(figsize=(5, 3))
plt.plot(wavelengthlist, slopelist)
plt.xlabel('Wavelength(nm)')
plt.ylabel("Unit absorption spectrum(ppm*m-1)")
# plt.ylim(-0.6, 0.05)
plt.grid(True)
# 显示图表
plt.show()

# export the unit absorption spectrum result to a txt file
with open('../AHSI_MF/unit_absorption_spectrum.txt', 'w') as output:
    for index in (range(len(wavelengthlist))):
        output.write(str(wavelengthlist[index])+' '+str(slopelist[index])+'\n')

