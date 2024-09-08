import sys
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from scipy.integrate import trapz
from matplotlib import pyplot as plt
import numpy as np
from MyFunctions import needed_function as nf
from MyFunctions import AHSI_data as ad

# Description: compare the simulated radiance with different methane concentration profiles 

# read the simulated radiance data
radiance_path1 = "C:\\PcModWin5\\Usr\\EMIT.fl7"
radiance_path2 = "C:\\PcModWin5\\Usr\\EMIT_methane.fl7"
radiance_path3 = "C:\\PcModWin5\\Usr\\EMIT_methane_2.fl7"

emit_channel_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\EMIT_channels.npz"
emit_bands,convoluved_radiance1 = nf.get_simulated_satellite_radiance(radiance_path1,emit_channel_path,1500,2500)
_,convoluved_radiance2 = nf.get_simulated_satellite_radiance(radiance_path2,emit_channel_path,1500,2500)
_,convoluved_radiance3 = nf.get_simulated_satellite_radiance(radiance_path3,emit_channel_path,1500,2500)


# draw the plot of the convolved radiance
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)

ln1 = ax1.plot(emit_bands, convoluved_radiance1, label='EMIT origianl methane profile', color='red', alpha=0.6)
ln2 = ax1.plot(emit_bands, convoluved_radiance2, label='EMIT 1.5*enhanced methane profile', color='blue', alpha=0.6)
ln2_5 = ax1.plot(emit_bands, convoluved_radiance3, label='EMIT 2*enhanced methane profile', color='black', alpha=0.6)

ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)')
ax1.set_xlim(1400, 2600)
ax1.grid(True)

portation1 = convoluved_radiance2/convoluved_radiance1
portation2 = convoluved_radiance3/convoluved_radiance1

ax1_left = ax1.twinx()
ln3 = ax1_left.plot(emit_bands,portation1, label='Methane extinction', color='black', linestyle='--', alpha=0.6)
ln4 = ax1_left.plot(emit_bands,portation2, label='Methane extinction', color='blue', linestyle='--', alpha=0.6)
ax1_left.set_ylabel('Extinction ratio')
ax1_left.set_ylim(0.5, 1)  # 根据实际透射率数据范围调整
ax1_left.set_xlim(1400, 2600)
plt.show()



# real radiance
# filepath = "C:\\Users\\RS\\Desktop\\GF5-02_李飞论文所用数据\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222_SW.tif"
# bands,real_spectrum = ad.get_calibrated_radiance(filepath,1500,2500)
# mean = np.mean(real_spectrum, axis=(1,2))

# fig,ax = plt.subplots()
# ax.plot(bands,mean,label='real radiance')
# plt.show()


