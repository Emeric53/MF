import math
import numpy as np
import matplotlib.pyplot as plt
total_radiance = []
original_radiance = []
wavelengthlist = []

# 获取文件路径
file_name = f"E:\\modtran5.2.6\\TEST\\EMIT_tp5\\Emit_0.chn"
with open(file_name, "r") as file:
    rawdata = file.readlines()[5:]
wavelengthlist = []
radiancelist = []
for i in rawdata:
    wavelength = float(i[2:12])
    wavelengthlist.append(wavelength)
    radiance = i[35:47]
    # radiance单位为 uW/sr cm-2 nm
    radiance = float(radiance) * 1000000000000000000000000
    # 对radiance取对数
    radiance_log = math.log(radiance, math.e)
    radiancelist.append(radiance_log)
total_radiance.append(radiancelist)

# 读取100个模拟结果文件
for index in range(7):
    # 获取文件路径
    file_name = f"E:\\modtran5.2.6\\TEST\\EMIT_tp5\\Emit_{int(math.pow(2,index)*1000)}.chn"
    with open(file_name, "r") as file:
        rawdata = file.readlines()[5:]
    wavelengthlist = []
    radiancelist = []
    for i in rawdata:
        wavelength = float(i[2:12])
        wavelengthlist.append(wavelength)
        radiance = i[35:47]
        #radiance单位为 uW/sr cm-2 nm
        radiance = float(radiance)*1000000000000000000000000
        #对radiance取对数
        radiance_log = math.log(radiance, math.e)
        radiancelist.append(radiance_log)
    total_radiance.append(radiancelist)
# 将radiance谱转为numpy数组并进行转置
total_radiance = np.array(total_radiance)
total_radiance = np.transpose(total_radiance)
print(total_radiance.shape)
print(len(wavelengthlist))
#构造用于线性回归的x轴数组
x = [0, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
x = np.array(x)
x = x/8000

slopelist = []
# 对每个波段进行线性回归拟合
for index in range(len(wavelengthlist)):
    data = total_radiance[index,:]
    print(data)
    # 使用polyfit函数进行线性回归拟合
    slope, intercept = np.polyfit(x,data ,1)
    slopelist.append(slope)
print(slopelist)
slopelist = np.array(slopelist)
print(slopelist.min())
# 绘制单位吸收光谱
plt.plot(wavelengthlist,slopelist)
plt.xlabel('Wavelength(nm)')
plt.ylabel("Unit absorption spectrum(ppm*m)-1")
plt.xlim(2100, 2500)
# plt.ylim(-0.6, 0.05)
plt.grid(True)
plt.show()

#  将单位吸收光谱写入文件
with open('New_EMIT_unit_absorption_spectrum.txt', 'w') as output:
    for index in (range(len(wavelengthlist))):
        output.write(str(wavelengthlist[index])+' '+str(slopelist[index])+'\n')
