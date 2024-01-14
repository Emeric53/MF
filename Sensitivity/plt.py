import matplotlib.pyplot as plt
import pandas as pd

# 从文本文件中读取数据
data_file = "../EMIT_MF/slope.txt"
x = []  # 存储 x 轴数据
y = []  # 存储 y 轴数据

with open(data_file, "r") as file:
    with open('../EMIT_MF/frequency.txt', 'r') as file2:
        for line in file:
            line = line.strip().split()  # 根据数据分隔符（空格、逗号等）分割数据
            y.append(float(line[0]))  # 假设第一列是 x 数据
        for line in file2:
            x.append((1/float(line)*10000000))

# # 从文本文件中读取数据
# data_file = "EMIT_unit_absorption_spectrum.txt"
# x = []  # 存储 x 轴数据
# y = []  # 存储 y 轴数据
#
# with open(data_file, "r") as file:
#     for line in file:
#         line = line.strip().split()  # 根据数据分隔符（空格、逗号等）分割数据
#         y.append(float(line[0]))  # 假设第一列是 x 数据
#     with open("C:\\Users\\RS\\Documents\\DataProcessing\\data\\EMIT\\properties.txt", "r") as file2:
#         for line in file2:
#             parts = line.split()  # 分割每行的数据
#             if len(parts) == 4:
#                 value = float(parts[3])
#                 x.append(value)

new_x = []
new_y = []
new_x.append(x[-1])
new_y.append(y[-1])
for i in range(len(x),-1,-1):
    if x[i-1]-new_x[-1] > 7.5:
        new_x.append(x[i])
        new_y.append(y[i]*10000)
# 创建图表
print(new_x)

# # plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.plot(new_x, new_y)

data = pd.DataFrame({'wl': new_x, 'value': new_y})
filtered_data = data[(data['wl'] >= 2100) & (data['wl'] <= 2500)]
plt.plot(filtered_data['wl'], filtered_data['value'])
plt.xlabel('Wavelength(nm)')
plt.ylabel("Unit absorption spectrum(ppm*m-1 )")

# 显示图表
plt.show()

