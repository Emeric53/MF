import numpy as np

"""
   该代码用于计算 modtran默认气象模型的廓线数据
"""

# model_type: 1-6
#   Tropical Model (15 degrees North)
#   Midlatitude Summer (45 degrees North, July)
#   Midlatitude Winter (45 degrees North, January)
#   Subarctic Summer (60 degrees North, July)
#   Subarctic Winter (60 degrees North, January)
#   1976 US Standard Atmosphere

# parameter: alt, pressure, temp, h2o, co2, o3, n2o, co, ch4, o2, density, no, so2

def get_profile(model_type, parameter):
    data_list = []
    with open('./Needed_data/mlatmb.f', "r") as file:
        index = get_index(model_type, parameter)
        lines = file.readlines()[index:index+10]
        for line in lines:
            # 去除每行数据中的"&"字符，然后从行中提取数据部分，并将数据分割成单个数值
            line = line.replace("&", "")
            line = line.replace("/", "")  # 去除斜杠
            data_parts = line.strip().split(",")
            values = [float(part.strip()) for part in data_parts if part.strip()]
            for value in values:
                data_list.append(value)
        dataarray = np.array(data_list)
        if parameter == "pressure":
            dataarray = np.exp(dataarray)*100
    return dataarray


def get_index(model_type, parameter):
    if parameter == "alt":
        start_index = 43
    elif parameter == "pressure":
        start_index = 56+(model_type-1)*11
    elif parameter == "temp":
        start_index = 124+(model_type-1)*11
    elif parameter == "h20":
        start_index = 192+(model_type-1)*11
    elif parameter == "co2":
        start_index = 260+(model_type-1)*11
    elif parameter == "o3":
        start_index = 328+(model_type-1)*11
    elif parameter == "n2o":
        start_index = 396+(model_type-1)*11
    elif parameter == "co":
        start_index = 464+(model_type-1)*11
    elif parameter == "ch4":
        start_index = 532+(model_type-1)*11
    elif parameter == "o2":
        start_index = 600+(model_type-1)*11
    elif parameter == "density":
        start_index = 668+(model_type-1)*11
    elif parameter == "no":
        start_index = 736
    elif parameter == "so2":
        start_index = 749
       
    return start_index


def trapezoidal_integration(x, y):
    integral = np.trapz(y, x)
    return integral


def calculate_molar_density(pressure, temperature):
    R = 8.314  # 理想气体常数，单位为 J/(mol·K)
    molar_density = pressure / (R * temperature)
    return molar_density


def molar_to_number(molar_density):
    N_A = 6.022e23  # 阿伏伽德罗常数，单位为个/mol
    number_density = molar_density * N_A
    return number_density


def nubmer_to_molar(number_density):
    N_A = 6.022e23  # 阿伏伽德罗常数，单位为个/mol
    molar_density = number_density / N_A
    return molar_density


methane_profile = get_profile(1, "ch4")
altitude_profile = get_profile(1, "alt")
pressure_profile = get_profile(1, "pressure")
temp_profile = get_profile(1, "temp")
dryair_molar_profile= nubmer_to_molar(get_profile(1, "density"))

print(dryair_molar_profile)
print(calculate_molar_density(pressure_profile, temp_profile))

methane_molar_profile = dryair_molar_profile * methane_profile

# 计算干空气和甲烷柱浓度
air_column_concentration = trapezoidal_integration(altitude_profile, dryair_molar_profile)
methane_column_concentration = trapezoidal_integration(altitude_profile, methane_molar_profile)
mixing_ratio = methane_column_concentration/air_column_concentration
print(mixing_ratio)

# 基于缩放因子 获得新的甲烷廓线
new_methane_profile = methane_profile * 1.9/mixing_ratio
print(new_methane_profile)
np.save( "./Needed_data/model1_methane1900ppb.npy",new_methane_profile)
