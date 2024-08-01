import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import sys
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from Tools.needed_function import build_lookup_table,save_lookup_table,load_lookup_table,lookup_spectrum,generate_transmittance_cube
# 构建查找表


if __name__ == "__main__":
    
    # # 构建查找表
    # enhancements= np.linspace(0,20000,401)
    # wavelengths, lookup_table = build_lookup_table(enhancements)

    # # 保存查找表到文件
    # save_lookup_table('./Needed_data/AHSI_trans_lookup_table.npz', wavelengths, lookup_table)

    # 从文件加载查找表
    loaded_wavelengths, loaded_lookup_table = load_lookup_table('./Needed_data/AHSI_trans_lookup_table.npz')
    _,approx_spectrum = lookup_spectrum(0, loaded_wavelengths,loaded_lookup_table, 1500,2500)  
