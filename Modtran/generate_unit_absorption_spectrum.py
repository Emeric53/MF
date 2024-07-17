import numpy as np
import math
import matplotlib.pyplot as plt
from tools.needed_function import get_simulated_satellite_radiance


# 处理流程分为三步
# 1.读取模拟数据
# 2.提取单位吸收光谱
# 3.绘制单位吸收光谱并进行保存


# 1.从modtran模拟结果中提取单位吸收光谱

ahsi_channels_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
emit_channels_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\EMIT_channels.npz"
lower_wavelength=1200
upper_wavelength=2500

# 读取
interval_range = np.array([5000,7500,10000])
enhance_range = np.arange(0,40.1,0.1)
for interval in interval_range:
    total_radiance = []
    for i in enhance_range:
        filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_{int(i*500)}_ppmm_interval{int(interval)}_tape7.txt"
        try:
            bands,convoluved_radiance = get_simulated_satellite_radiance(filepath,ahsi_channels_path,lower_wavelength,upper_wavelength)
            log_convoluved_radiance = [math.log(i,math.e) for i in convoluved_radiance]
            total_radiance.append(log_convoluved_radiance)
        except Exception as e:
    # Code that runs for any other exception
            print(f"An unexpected error occurred: {e}")
            print(filepath)
    # 将数据转为numpy数组,并转置,使得第一个维度为波长,第二个维度为模拟中的variable(例如methane,water vapor等)
    total_radiance = np.transpose(np.array(total_radiance))

    # 2.提取单位吸收光谱
    slopelist = []
    for index,data in enumerate(total_radiance):
        # 使用polyfit函数进行线性回归拟合
        slope,_ = np.polyfit(enhance_range*500,data,1)
        slopelist.append(slope)
    # export the unit absorption spectrum result to a txt file
    with open(f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum_interval{interval}.txt", 'w') as output:
        for index,data in enumerate(total_radiance):
            output.write(str(bands[index])+' '+str(slopelist[index])+'\n')

# 3.绘制单位吸收光谱并进行保存
plt.figure(figsize=(5, 3))
plt.plot(bands, slopelist)
plt.xlabel('Wavelength(nm)')
plt.ylabel("Unit absorption spectrum(ppm*m-1)")
# plt.ylim(-0.6, 0.05)
plt.grid(True)
# 显示图表
plt.show()
