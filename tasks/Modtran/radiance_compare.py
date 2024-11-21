from matplotlib import pyplot as plt
import numpy as np

import sys

sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from utils.satellites_data import general_functions as gf
from utils import satellites_data as sd
# Description: compare the simulated radiance with different methane concentration profiles


def draw_radiance(ax, radiance_path: str, satellite_name: str):
    channel_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\data\\satellite_channels\\{satellite_name}_channels.npz"

    bands, convoluved_radiance = gf.get_simulated_satellite_radiance(
        radiance_path, channel_path, 1500, 2500
    )
    ax.plot(bands, convoluved_radiance, label=f"{satellite_name} radiance", alpha=0.6)
    return bands, convoluved_radiance


def emit_radiance_compare():
    # draw the plot of the convolved radiance
    ax1 = plt.subplot2grid(
        (3, 3),
        (0, 0),
        colspan=3,
        rowspan=2,
    )

    # read the simulated radiance data
    radiance_path1 = "C:\\PcModWin5\\Usr\\EMIT.fl7"
    radiance_path2 = "C:\\PcModWin5\\Usr\\EMIT_methane.fl7"
    radiance_path3 = "C:\\PcModWin5\\Usr\\EMIT_methane_2.fl7"

    bands, randiance = gf.read_simulated_radiance(radiance_path1)
    indice = np.where((bands >= 1500) & (bands <= 2500))
    bands = bands[indice]
    randiance = randiance[indice]
    ax1.plot(bands, randiance, label="EMIT radiance", alpha=0.6)

    emit_bands, convoluved_radiance1 = draw_radiance(ax1, radiance_path1, "EMIT")
    emit_bands, convoluved_radiance2 = draw_radiance(ax1, radiance_path2, "EMIT")
    emit_bands, convoluved_radiance3 = draw_radiance(ax1, radiance_path3, "EMIT")

    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)")
    ax1.set_xlim(1400, 2600)
    ax1.grid(True)
    plt.show()

    # portation1 = convoluved_radiance2 / convoluved_radiance1
    # portation2 = convoluved_radiance3 / convoluved_radiance1

    # ax1_left = ax1.twinx()
    # ax1_left.plot(
    #     bands,
    #     portation1,
    #     label="Methane extinction",
    #     color="black",
    #     linestyle="--",
    #     alpha=0.6,
    # )
    # ax1_left.plot(
    #     emit_bands,
    #     portation2,
    #     label="Methane extinction",
    #     color="blue",
    #     linestyle="--",
    #     alpha=0.6,
    # )
    # ax1_left.set_ylabel("Extinction ratio")
    # ax1_left.set_ylim(0.5, 1)  # 根据实际透射率数据范围调整
    # ax1_left.set_xlim(1400, 2600)
    # plt.show()


# real radiance
# filepath = "C:\\Users\\RS\\Desktop\\Lifei_essay_data\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222_SW.tif"
# bands, real_spectrum = sd.GF5B_data.get_calibrated_radiance(filepath, 1500, 2500)
# mean = np.mean(real_spectrum, axis=(1, 2))

# fig, ax = plt.subplots()
# ax.plot(bands, mean, label="real radiance")
# plt.show()


emit_radiance_compare()
