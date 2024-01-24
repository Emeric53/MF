"""该代码用于分析 SensitivityAnalysis 文件夹中的数据，绘制出不同参数对应的radiance差值图 找出对EMIT传感器端radiance影响较大的参数"""
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# 指定目录路径
directory = "C:\\Users\\RS\\Desktop\\modtran5.2.6\\TEST\\SensitivityAnalysis"

# 获取目录中所有文件的名称
files = [os.path.abspath(file) for file in glob.glob(os.path.join(directory, '*.chn'))]
filename = [os.path.basename(file) for file in glob.glob(os.path.join(directory, '*.chn'))]

# 创建一个空字典，用于存储不同的radiance 数组
data_dict = {}

# 遍历文件名列表，为每个文件名创建一个空数组
for name in filename:
    data_dict[name] = []

# 创建波长和radiance数组
wavelengthlist = []
radiance = []

# 读取数据 填入波长数据
with open(files[0], 'r') as file:
    rawdata = file.readlines()[5:]
for i in rawdata:
    wavelength = float(i.split()[0])
    wavelengthlist.append(wavelength)

# 遍历文件列表，读取radiance数据
for index in range(len(files)):
    with open(files[index], 'r') as file:
        rawdata = file.readlines()[5:]
    for i in rawdata:
        data_dict[filename[index]].append(float(i.split()[3])*1000000)


# 绘制数据
for name, value in data_dict.items():
    if not name == 'original.chn':
        if not name =="albedo_1.chn":
            plt.plot(wavelengthlist[-60:], np.array(value[-60:])-np.array(data_dict['original.chn'][-60:]))

# 设置图形属性
plt.xlabel('Wavelength(nm)')
plt.ylabel("Radiance(uW/sr cm-2 nm)")
plt.grid(True)
legend = plt.legend(filename, loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)  # 显示图例
# plt.savefig('legend1.png', bbox_inches='tight')
plt.setp(legend.get_texts(), fontsize="small")  # 设置图例文本的字体大小
plt.show()
