import numpy as np
import math
import csv
#"C:\Users\RS\Desktop\mlatmb.f"
# 打开.f文件并读取指定行的数据
data = []  # 用于存储提取的甲烷 ppmv 数据
heights = []  # 用于存储高度数据
pressure = [] #存储气压数据
temp = [] #存储温度数据
formula = [] #用于存储分子数

with open("C:/Users/RS/Desktop/mlatmb.f", "r") as file:
    lines = file.readlines()[587:597]  # 读取第533到542行的数据

    for line in lines:
        # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
        line = line.replace("&", "")
        line = line.replace("/", "")  # 去除斜杠
        data_parts = line.strip().split(",")
        values = [float(part.strip()) for part in data_parts if part.strip()]
        data.extend(values)

with open("C:/Users/RS/Desktop/mlatmb.f", "r") as file:
    lines = file.readlines()[111:121]  # 读取第533到542行的数据

    for line in lines:
        # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
        line = line.replace("&", "")
        line = line.replace("/", "")  # 去除斜杠
        data_parts = line.strip().split(",")
        values = [math.exp(float(part.strip())) for part in data_parts if part.strip()]
        pressure.extend(values*100)

with open("C:/Users/RS/Desktop/mlatmb.f", "r") as file:
    lines = file.readlines()[179:189]  # 读取第533到542行的数据

    for line in lines:
        # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
        line = line.replace("&", "")
        line = line.replace("/", "")  # 去除斜杠
        data_parts = line.strip().split(",")
        values = [float(part.strip()) for part in data_parts if part.strip()]
        temp.extend(values)

with open("C:/Users/RS/Desktop/mlatmb.f", "r") as file:
    lines = file.readlines()[43:53]  # 读取第533到542行的数据

    for line in lines:
        # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
        line = line.replace("&", "")
        line = line.replace("/", "")  # 去除斜杠
        data_parts = line.strip().split(",")
        values = [float(part.strip()) for part in data_parts if part.strip()]
        heights.extend(values)

for p, t in zip(pressure, temp):
    formula.append(7.244*math.pow(10, 10)*p/t)

numberdensity = []
for p, t in zip(data, formula):
    numberdensity.append(p*t)
print(numberdensity)

# numberdensity_1 = []
# data_1 = [ element*1.1 for element in data]
# for p, t in zip(data_1, formula):
#     numberdensity_1.append(p*t)
# print(numberdensity_1)

def trapezoidal_integration(x, y):
    integral = np.trapz(y, x)
    return integral

# 计算甲烷柱浓度
column_density = trapezoidal_integration(heights, numberdensity)
print(column_density*100000)
print(column_density/heights[-1])


