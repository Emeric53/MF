"""该代码用于计算 modtran默认气象模型的廓线数据"""
import math

import numpy as np

# "C:\Users\RS\Desktop\mlatmb.f"
# 打开.f文件并读取指定行的数据
data = []  # 用于存储提取的 气体 数据
heights = []  # 用于存储高度数据
pressure = []  # 用于存储高度数据
temp = []  # 用于存储高度数据
number = []  # 用于存储分子数
m_number = []  # 用于存储分子数

with open("C:/Users/RS/Desktop/mlatmb.f", "r") as file:
    lines = file.readlines()[587:597]
    for line in lines:
        # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
        line = line.replace("&", "")
        line = line.replace("/", "")  # 去除斜杠
        data_parts = line.strip().split(",")
        values = [float(part.strip()) for part in data_parts if part.strip()]
        data.append(values)

with open("C:/Users/RS/Desktop/mlatmb.f", "r") as file:
    lines = file.readlines()[111:121]
    for line in lines:
        # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
        line = line.replace("&", "")
        line = line.replace("/", "")  # 去除斜杠
        data_parts = line.strip().split(",")
        values = [math.exp(float(part.strip())) for part in data_parts if part.strip()]
        pressure.append(values)

with open("C:/Users/RS/Desktop/mlatmb.f", "r") as file:
    lines = file.readlines()[179:189]
    for line in lines:
        # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
        line = line.replace("&", "")
        line = line.replace("/", "")  # 去除斜杠
        data_parts = line.strip().split(",")
        values = [float(part.strip()) for part in data_parts if part.strip()]
        temp.append(values)

# 计算分子摩尔数？
for p, t in zip(pressure, temp):
    number.append(p / (t * 8.314))

for p, t in zip(number, data):
    m_number.append(p * t)
print(m_number)

with open("C:/Users/RS/Desktop/mlatmb.f", "r") as file:
    lines = file.readlines()[43:53]  # 读取第533到542行的数据

    for line in lines:
        # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
        line = line.replace("&", "")
        line = line.replace("/", "")  # 去除斜杠
        data_parts = line.strip().split(",")
        values = [float(part.strip()) for part in data_parts if part.strip()]
        heights.extend(values)
# 打印提取的值
print("提取的值：", data)
print("提取的值：", heights)


def trapezoidal_integration(x, y):
    integral = np.trapz(y, x)
    return integral


# 计算甲烷柱浓度
air_column_concentration = trapezoidal_integration(heights, number)
methane_column_concentration = trapezoidal_integration(heights, m_number)

factor = 2.35 / (methane_column_concentration / air_column_concentration)
newdata = []
for i in data:
    newdata.append(i * factor)
print(factor)
print(newdata)
print("甲烷柱浓度（ppm）：", methane_column_concentration / air_column_concentration)

csv_file_name = "../2.35_US_1987_CH4_profile.txt"

# 打开文件并写入CSV数据
with open(csv_file_name, mode='w') as file:
    for item in newdata:
        file.write(str(item) + '\n')

csv_file_name = "../altitude.txt"

# 打开文件并写入CSV数据
with open(csv_file_name, mode='w') as file:
    for item in heights:
        file.write(str(item) + '\n')
