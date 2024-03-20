"""该代码用于计算 modtran默认气象模型的廓线数据"""
import math
import numpy as np

def trapezoidal_integration(x, y):
    integral = np.trapz(y, x)
    return integral

data = []  # 用于存储提取的气体数据 此处为体积混合比
heights = []  # 用于存储高度数据
pressure = []  # 用于存储气压数据
temp = []  # 用于存储温度数据
number = []  # 用于存储干空气分子数
m_number = []  # 用于存储特定气体分子数


#读取甲烷的体积混合比数据
with open("mlatmb.f", "r") as file:
    lines = file.readlines()[587:597]
    for line in lines:
        # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
        line = line.replace("&", "")
        line = line.replace("/", "")  # 去除斜杠
        data_parts = line.strip().split(",")
        values = [float(part.strip()) for part in data_parts if part.strip()]
        for value in values:
            data.append(value)

#读取气压数据数据
with open("mlatmb.f", "r") as file:
    lines = file.readlines()[111:121]
    for line in lines:
        # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
        line = line.replace("&", "")
        line = line.replace("/", "")  # 去除斜杠
        data_parts = line.strip().split(",")
        values = [math.exp(float(part.strip())) for part in data_parts if part.strip()]
        for value in values:
            pressure.append(value)

#读取温度数据
with open("mlatmb.f", "r") as file:
    lines = file.readlines()[179:189]
    for line in lines:
        # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
        line = line.replace("&", "")
        line = line.replace("/", "")  # 去除斜杠
        data_parts = line.strip().split(",")
        values = [float(part.strip()) for part in data_parts if part.strip()]
        for value in values:
            temp.append(value)

with open("mlatmb.f", "r") as file:
    lines = file.readlines()[43:53]  # 读取第533到542行的数据
    for line in lines:
        # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
        line = line.replace("&", "")
        line = line.replace("/", "")  # 去除斜杠
        data_parts = line.strip().split(",")
        values = [float(part.strip()) for part in data_parts if part.strip()]
        heights.extend(values)

# 计算空气分子摩尔数
for p, t in zip(pressure, temp):
    number.append(p / (t * 8.314))

#计算甲烷分子摩尔数
for p, t in zip(number, data):
    m_number.append(p * t)

# 计算干空气和甲烷 柱浓度
air_column_concentration = trapezoidal_integration(heights, number)
methane_column_concentration = trapezoidal_integration(heights, m_number)

# 计算需要的 甲烷柱浓度所需要的 缩放因子
factor = 2 / (methane_column_concentration / air_column_concentration)

# 基于缩放因子 获得新的甲烷廓线
newdata = []
for i in data:
    newdata.append(i * factor)

# 打开文件并写入CSV数据
csv_file_name = "2ppm_US_1987_CH4_profile.txt"
with open(csv_file_name, mode='w') as file:
    for item in newdata:
        file.write(str(item) + '\n')

# csv_file_name = "altitude.txt"
# # 打开文件并写入CSV数据
# with open(csv_file_name, mode='w') as file:
#     for item in heights:
#         file.write(str(item) + '\n')
