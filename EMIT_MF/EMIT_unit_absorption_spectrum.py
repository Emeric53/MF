""" 基于模拟的 radiance 计算得到  EMIT传感器的单位吸收光谱 """
import math
import numpy as np
# 创建一个存储数据的列表
total_radiance = []

# # 循环读取文件并存储数据
for j in range(1, 101):  # 读取100个文件
    file_name = f"C:\\PcModWin5\\Bin\\batch\\EMIT_{j * 100}_tape7.txt"  # 生成文件名
    with open(file_name, "r") as file:
        rawdata = file.readlines()[11:3577]
    final = {}
    frequencylist = []
    wavelengthlist = []
    radiancelist = []
    for i in rawdata:
        frequency = i[0:8]
        frequency = float(frequency)
        wavelength = 1 / frequency
        frequencylist.append(frequency)
        wavelengthlist.append(wavelength)
        radiance = i[97:107]
        radiance = float(radiance)
        radiance = radiance / math.pow(wavelength, 2) / 10000 * 1000
        print(radiance)
        radiance_log = math.log(radiance)
        radiancelist.append(radiance_log)
    total_radiance.append(radiancelist)

total_radiance = np.array(total_radiance)
total_radiance = np.transpose(total_radiance)
print(total_radiance)

slopelist = []
for index in range(len(frequencylist)):
    data = total_radiance[index]
    x = np.arange(len(data))
    x = x * 100

    # 使用polyfit函数进行线性回归拟合
    slope, intercept = np.polyfit(x, data, 1)
    if slope > 0:
        slope = 0
    slopelist.append(slope)

with open('frequency.txt', 'w') as file:
    for i in frequencylist:
        file.write(str(i) + '\n')
with open('log_radiance_list.txt', 'w') as file:
    for i in total_radiance:
        file.write(str(i) + '\n')
with open('slope.txt', 'w') as file:
    for i in slopelist:
        file.write(str(i) + '\n')

# with open('frequency.txt', 'r') as file:
#     with open('slope.txt', 'r') as file2:
#         with open('unit_absorption_spectrum.txt', 'w') as output:
#             frequency = file.readlines()
#             slope = file2.readlines()
#             for index in reversed(range(len(frequency))):
#                 freq = (1 / float(frequency[index])) * 10000000
#                 s = float(slope[index])
#                 output.write(str(freq) + ' ' + str(s) + '\n')
