from matplotlib import pyplot as plt
from MyFunctions.needed_function import get_simulated_satellite_transmittance
# Description: This script demonstrates how to convolve a spectrum with a Gaussian response function.

transmittance_path1 = r"C:\\PcModWin5\\Usr\\Trans_1.fl7"
transmittance_path2 = r"C:\\PcModWin5\\Usr\\Trans_1.5.fl7"
transmittance_path3 = r"C:\\PcModWin5\\Usr\\Trans_2.fl7"
emit_channel_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\EMIT_channels.npz"


bands,convoluved_trans1 = get_simulated_satellite_transmittance(transmittance_path1,emit_channel_path,1500,2500)
_,convoluved_trans2 = get_simulated_satellite_transmittance(transmittance_path2,emit_channel_path,1500,2500)
_,convoluved_trans3 = get_simulated_satellite_transmittance(transmittance_path3,emit_channel_path,1500,2500)

# get the enhanced transmittance due to the methane
enhanced_trans1 = convoluved_trans2/convoluved_trans1
enhanced_trans2 = convoluved_trans3/convoluved_trans1

# 顶部大图
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
ln1 = ax1.plot(bands, enhanced_trans1, label='1.5*enhanced transmittance', color='gray', alpha=0.6)
ln2 = ax1.plot(bands, enhanced_trans2, label='2*enhanced transmittance', color='blue', alpha=0.6)
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Transmittance')
ax1.set_ylim(0.8,1)
ax1.set_xlim(1000, 2500)
ax1.grid(True)

plt.show()
