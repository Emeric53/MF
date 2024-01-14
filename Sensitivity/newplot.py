import matplotlib.pyplot as plt

# 读取波长数据
with open("C:\\Users\\RS\\Desktop\\2.txt", 'r') as file:
    wavelengths = [float(line.strip()) for line in file]

# 读取对应的值
with open("C:\\Users\\RS\\Desktop\\1.txt", 'r') as file:
    values = [float(line.strip()) for line in file]

# 筛选出波长范围在2100到2500之间的数据
selected_wavelengths = []
selected_values = []
for i in range(len(wavelengths)):
    if 2100 <= wavelengths[i] <= 2500:
        selected_wavelengths.append(wavelengths[i])
        selected_values.append(values[i])
# 创建图表
plt.plot(selected_wavelengths, selected_values)
plt.title('')
plt.xlabel('波长')
plt.ylabel('Radiance')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
# 显示图表
plt.show()