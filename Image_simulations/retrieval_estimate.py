import sys 
import os
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
# from Emissionrate_estimate import Estimate_emission_rate
import numpy as np
from tools import needed_function
plumefolder = r"C:\Users\RS\VSCode\matchedfiltermethod\Needed_data\plumes"
names = ["wetland","urban","grassland","desert"]
for stability in ['D','E']:
    for windspeed in [2,4,6,8,10]:
        for name in names:
            plume_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\plumes\\gaussianplume_1000_{windspeed}_stability_{stability}.npy"
            plume = np.load(plume_path)
            high_concentration_mask = plume > 100
            molar_mass_CH4 = 16.04  # 甲烷的摩尔质量，g/mol
            molar_volume_STP = 0.0224  # 摩尔体积，m^3/mol at STP
            emission = np.sum(plume[high_concentration_mask])*900*(molar_mass_CH4/molar_volume_STP) * 1e-6
            tiff_path = f"I:\\simulated_images_nonoise\\result\\{name}_q_1000_u_{windspeed}_stability_{stability}.tif"
            result = needed_function.read_tiff(tiff_path)
            retrieval_emission = np.sum(result[0,high_concentration_mask])*900*(molar_mass_CH4/molar_volume_STP) * 1e-6
            print("the simulated image name: "+ os.path.basename(tiff_path))
            print("simulated emission is "+ str(emission))  
            print("retrieved emission is "+ str(retrieval_emission))      
