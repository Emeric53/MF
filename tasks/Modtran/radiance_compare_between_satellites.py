from matplotlib import pyplot as plt
import numpy as np

import sys

sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src")
from utils.satellites_data import general_functions as gf
from utils import satellites_data as sd
# Description: compare the simulated radiance with different methane concentration profiles


def draw_radiance(ax, radiance_path: str, satellite_name: str):
    channel_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\satellite_channels\\{satellite_name}_channels.npz"

    bands, convoluved_radiance = gf.get_simulated_satellite_radiance(
        radiance_path, channel_path, 1500, 1800
    )
    ax.plot(bands, convoluved_radiance, label=f"{satellite_name} radiance", alpha=0.6)
    return bands, convoluved_radiance


def radiance_compare():
    # draw the plot of the convolved radiance
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)

    # read the simulated radiance data
    original_radiance_path = "C:\\PcModWin5\\Bin\\batch\\sensitivity_base_tape7.txt"

    satellite_bands, convoluved_radiance = draw_radiance(
        ax1, original_radiance_path, "AHSI"
    )
    satellite_bands, convoluved_radiance = draw_radiance(
        ax1, original_radiance_path, "EMIT"
    )
    satellite_bands, convoluved_radiance = draw_radiance(
        ax1, original_radiance_path, "EnMAP"
    )
    satellite_bands, convoluved_radiance = draw_radiance(
        ax1, original_radiance_path, "PRISMA"
    )

    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)")
    ax1.set_xlim(1400, 2600)
    ax1.grid(True)
    plt.show()


# real radiance
filepath = "C:\\Users\\RS\\Desktop\\GF5-02_李飞论文所用数据\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222_SW.tif"
bands, real_spectrum = sd.GF5B_data.get_calibrated_radiance(filepath, 1500, 2500)
mean = np.mean(real_spectrum, axis=(1, 2))

fig, ax = plt.subplots()
ax.plot(bands, mean, label="real radiance")
plt.show()
