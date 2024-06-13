import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
with open(r"C:\PcModWin5\Bin\channels.out", 'r') as radiance:
    radiance = radiance.readlines()[5:-1]
    radiance = [float(data[60:73]) for data in radiance]
    wavelength = np.linspace(410,900,50)

for i,j in zip(radiance,wavelength):
    # print(i*10000)
    print(i*j*(10e-9)/(6.626*(10e-34)*3*(10e8)))
print("     \n    \n")
for i,j in zip(radiance,wavelength):
    print(i*10000)
    # print(i*j*(10e-9)/(6.626*(10e-34)*3*(10e8)))
# print(wavelength)
#
# fig, ax = plt.subplots()
# ax.plot(wavelength,radiance)
#
# # 设置 y 轴刻度格式
# formatter = ticker.ScalarFormatter(useMathText=True)
# formatter.set_powerlimits((0, 0))
# ax.yaxis.set_major_formatter(formatter)
#
# # 设置 y 轴标签
# ax.set_ylabel("radiance unit: W Sr-1 cm-2 nm-1")
# plt.xlabel("Wavelength (nm)")
# plt.title("radiance simulation for case 4")
#
# plt.show()

