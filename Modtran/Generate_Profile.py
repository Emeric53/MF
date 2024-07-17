import math
import numpy as np
"""
   该代码用于计算 modtran默认气象模型的廓线数据
"""

def trapezoidal_integration(x, y):
    integral = np.trapz(y, x)
    return integral


def get_data(data_list, index=[]):
    with open('./Needed_data/mlatmb.f', "r") as file:
        lines = file.readlines()[index[0]:index[1]]
        for line in lines:
            # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
            line = line.replace("&", "")
            line = line.replace("/", "")  # 去除斜杠
            data_parts = line.strip().split(",")
            values = [float(part.strip()) for part in data_parts if part.strip()]
            for value in values:
                data_list.append(value)


def find_indices(arr1, arr2):
    indices = []
    for element in arr2:
        index = arr1.index(element)
        indices.append(index)
    return indices

data = []  # 用于存储提取的气体数据 此处为体积混合比
heights = []  # 用于存储高度数据
log_pressure = []  # 用于存储气压数据
temp = []  # 用于存储温度数据
number = []  # 用于存储干空气分子数
m_number = []  # 用于存储特定气体分子数
# us 1987 
# get_data(data, [587, 597])
# get_data(log_pressure, [111, 121])
# get_data(temp, [179, 189])
# get_data(heights, [43, 53])
# pressure = [math.e**value for value in log_pressure]
# midlat summer
get_data(data, [543, 553])
get_data(log_pressure, [67, 77])
get_data(temp, [135, 145])
get_data(heights, [43, 53])
pressure = [math.e**value for value in log_pressure]
# 计算空气分子摩尔数
for p, t in zip(pressure, temp):
    number.append(p / (t * 8.314))

#计算甲烷分子摩尔数
for p, t in zip(number, data):
    m_number.append(p * t)

# 计算干空气和甲烷柱浓度
air_column_concentration = trapezoidal_integration(heights, number)
methane_column_concentration = trapezoidal_integration(heights, m_number)
mixing_ratio = methane_column_concentration/air_column_concentration

# 基于缩放因子 获得新的甲烷廓线
newdata = []
factor = 1.9/mixing_ratio
newdata.append(data[0] * factor)
for i in data:
    newdata.append(i * factor)

# #打开文件并写入CSV数据
# csv_file_name = "./Needed_data/Midlat_summer_1900.txt"
# with open(csv_file_name, mode='w') as file:
#     for item in newdata:
#         file.write(str(item) + '\n')

# csv_file_name = "./Needed_data/altitude.txt"
# # 打开文件并写入CSV数据
# with open(csv_file_name, mode='w') as file:
#     for item in heights:
#         file.write(str(item) + '\n')
