import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv(r"C:\Users\RS\Downloads\EnMAP_SWIR_Spectral_Bands.csv")

# 提取中心波长（CW）和全宽半高（FWHM）数据
cw_data = df["CW (nm)"].values
fwhm_data = df["FWHM (nm)"].values

# 将数据保存为npz文件
np.savez(
    r"C:\Users\RS\VSCode\matchedfiltermethod\src\data\satellite_channels\EnMAP_channels.npz",
    central_wvls=cw_data,
    fwhms=fwhm_data,
)
