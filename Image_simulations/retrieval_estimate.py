from Emissionrate_estimate import Emissionrate_estimate
import numpy as np

plumefolder = r"C:\Users\RS\VSCode\matchedfiltermethod\Needed_data\plumes"
for stability in ['D','E']:
    for windspeed in [2,4,6,8,10]:
        plume_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\plumes\\gaussianplume_1000_{windspeed}_stability_{stability}.npy"
        plume = np.load(plume_path)
        plume_mask =  np.where(plume>100,1,0)
        emission = plume[plume_mask].sum()*900
        print(emission)        
