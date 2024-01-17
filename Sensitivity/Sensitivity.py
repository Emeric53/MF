""" 敏感性分析 绘制plot """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

default = pd.read_csv("C:\\Users\\RS\\Desktop\\default.csv")
enhanced = pd.read_csv("C:\\Users\\RS\\Desktop\\albedo.csv")

with open("C:\\Users\\RS\\Desktop\\envi_plot.txt", 'r') as file:
    lines = file.readlines()
wl = []
value = []
for line in lines:
    part1 = line[2:14]
    wl.append(float(part1))
    part2 = line[15:].rstrip("\n")
    value.append(float(part2))

array1 = np.array(wl)
array2 = np.array(value)

# 原始数据
df = default[['FREQ(CM-1)', 'TOTAL RAD']]
# 将波数转换为wavelength（nm）
new_df = pd.DataFrame()
new_df['wavelength（nm）'] = 1 / (df['FREQ(CM-1)'])*10**7
new_df['radiance(uW/srcm-2nm)'] = df['TOTAL RAD']/new_df['wavelength（nm）']/new_df['wavelength（nm）']*10000000*1000000
# 选择wavelength范围为2000nm-2500nm的数据
filtered_data = new_df[(new_df['wavelength（nm）'] >= 2100) & (new_df['wavelength（nm）'] <= 2500)]


new_wl = []
new_value = []

emitwl = filtered_data['wavelength（nm）'].to_numpy()
emitvalue = filtered_data['radiance(uW/srcm-2nm)'].to_numpy()
print(emitwl)
print(array1)

for i in array1:
    index = np.abs(emitwl - i).argmin()
    new_wl.append(emitwl[index])
    new_value.append(emitvalue[index])
print(new_wl)
print(array1)
new_emit = pd.DataFrame({'wl': new_wl, 'value': new_value})

# 将数组合并为DataFrame
image = pd.DataFrame({'wl': array1, 'value': array2})

window_size = 5  # 调整窗口大小以控制平滑程度
new_emit['平滑透射率'] = new_emit['value'].rolling(window=window_size, min_periods=1).mean()

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(new_emit['wl'], new_emit['平滑透射率'], color='red' ,label=u"EMIT-地表反照率0.1")

# 对比数据
df = enhanced[['FREQ(CM-1)', 'TOTAL RAD']]
# 将波数转换为wavelength（nm）
df['wavelength（nm）'] = 1 / (df['FREQ(CM-1)'])*10**7
df['radiance(uW/srcm-2nm)'] = df['TOTAL RAD']/df['wavelength（nm）']/df['wavelength（nm）']*10000000*1000000
# 选择wavelength范围为2000nm-2500nm的数据
filtered_data = df[(df['wavelength（nm）'] >= 2100) & (df['wavelength（nm）'] <= 2500)]
new_wl = []
new_value = []

emitwl = filtered_data['wavelength（nm）'].to_numpy()
emitvalue = filtered_data['radiance(uW/srcm-2nm)'].to_numpy()
print(emitwl)
print(array1)

for i in array1:
    index = np.abs(emitwl - i).argmin()
    new_wl.append(emitwl[index])
    new_value.append(emitvalue[index])
print(new_wl)
print(array1)
new_emit = pd.DataFrame({'wl': new_wl, 'value': new_value})

# 将数组合并为DataFrame
image = pd.DataFrame({'wl': array1, 'value': array2})

window_size = 5  # 调整窗口大小以控制平滑程度
new_emit['平滑透射率'] = new_emit['value'].rolling(window=window_size, min_periods=1).mean()

plt.plot(new_emit['wl'], new_emit['平滑透射率'], color='blue' ,label=u"EMIT-地表反照率0.2")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.xlabel('Wavelength(nm)')
plt.ylabel('Radiance(uW/sr cm-2 nm)')
plt.title('EMIT-地表反照率敏感性分析')
plt.legend()
plt.grid(True)
plt.show()
