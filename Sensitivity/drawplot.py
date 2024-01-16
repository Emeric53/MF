import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# 指定目录路径
directory = "C:\\Users\\RS\\Desktop\\modtran5.2.6\\TEST\\SensitivityAnalysis"

# 获取目录中所有文件的名称
files = [os.path.abspath(file) for file in glob.glob(os.path.join(directory, '*.chn'))]
filename = [os.path.basename(file) for file in glob.glob(os.path.join(directory, '*.chn'))]

# 创建一个空字典，用于存储不同的数组
data_dict = {}

# 遍历文件名列表，为每个文件名创建一个空数组
for name in filename:
    data_dict[name] = []

wavelengthlist = []
radiance = []

with open(files[0], 'r') as file:
    rawdata = file.readlines()[5:]
for i in rawdata:
    wavelength = float(i.split()[0])
    wavelengthlist.append(wavelength)

for index in range(len(files)):
    with open(files[index], 'r') as file:
        rawdata = file.readlines()[5:]
    for i in rawdata:
        data_dict[filename[index]].append(float(i.split()[3])*1000000)
print(data_dict)

# 绘制数据
for name,value in data_dict.items():
    plt.plot(wavelengthlist, value)  # 使用wavelength作为x轴，data作为y轴
plt.xlabel('Wavelength(nm)')
plt.ylabel("Radiance(uW/sr cm-2 nm)")
plt.grid(True)
plt.legend(filename)  # 显示图例
# 显示图形
plt.show()
