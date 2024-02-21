# 该代码用于绘制单位吸收谱的plot
import matplotlib.pyplot as plt

wavelength = []
value = []

# 打开单位吸收光谱文件
with open("EMIT_unit_absorption_spectrum.txt", "r") as file:
    data = file.readlines()

# 分别读取波长和对应的吸收谱值
for i in range(len(data)):
    data[i] = data[i].split(" ")
    value.append(float(data[i][1]))
    wavelength.append(float(data[i][0]))

# 绘制单位吸收谱
plt.plot(wavelength, value)
plt.show()
