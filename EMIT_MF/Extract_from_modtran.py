import math
import numpy as np
import matplotlib.pyplot as plt
total_radiance = []
original_radiance = []
wavelengthlist = []

# 读取100个文件
for index in range(0, 101):
    file_name = f"C:\\Users\\RS\\Desktop\\modtran5.2.6\\TEST\\EMIT\\Emit_{index*100}.chn"  # 生成文件名
    with open(file_name, "r") as file:
        rawdata = file.readlines()[5:]
    final = {}
    wavelengthlist = []
    radiancelist = []
    for i in rawdata:
        wavelength = float(i[2:12])
        wavelengthlist.append(wavelength)
        radiance = i[35:47]
        #单位为 uW/sr cm-2 nm
        radiance = float(radiance)*1000000
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

plt.figure(figsize=(5, 3))
plt.plot(wavelengthlist, slopelist)
plt.xlabel('Wavelength(nm)')
plt.ylabel("Unit absorption spectrum(ppm*m-1)")
# plt.ylim(-0.6, 0.05)
plt.grid(True)
plt.show()

with open('EMIT_unit_absorption_spectrum.txt', 'w') as output:
    for index in (range(len(wavelengthlist))):
        output.write(str(wavelengthlist[index])+' '+str(slopelist[index])+'\n')
