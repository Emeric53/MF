import numpy as np
import os
import sys
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from MyFunctions.needed_function import read_tiff,get_tiff_files,export_to_tiff
from MatchedFilter.matched_filter import open_unit_absorption_spectrum,filter_and_slice,matched_filter
import matplotlib.pyplot as plt

def retrieval():
    # 甲烷浓度增强反演
    filelist, filenames = get_tiff_files('I:\\simulated_images_nonoise')
    uas_filepath = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum.txt"
    uas = open_unit_absorption_spectrum(uas_filepath)
    _, used_slice = filter_and_slice(uas[:, 0], 2150, 2500)
    # 获取目标反演窗口范围内的 单位吸收谱和辐亮度数据
    used_uas = uas[used_slice, 1]
    for i,filepath in enumerate(filelist):
        outputpath = os.path.join("I:\\simulated_images_nonoise\\result",filenames[i])
        if not os.path.exists(outputpath):
            data = read_tiff(filepath)[-len(used_uas):,:,:]
            average = np.mean(data,axis=(1,2))
            enhancement = matched_filter(data, used_uas,is_iterate=True, is_albedo= True, is_filter= False,is_columnwise=False)
            if enhancement is not None:
                export_to_tiff(enhancement,outputpath)
                print(filepath + " has been processed")


def quantification():
    # 甲烷排放量估计
    plumefolder = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\plumes"
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
                result = read_tiff(tiff_path)
                retrieval_emission = np.sum(result[0,high_concentration_mask])*900*(molar_mass_CH4/molar_volume_STP) * 1e-6
                print("the simulated image name: "+ os.path.basename(tiff_path))
                print("simulated emission is "+ str(emission))  
                print("retrieved emission is "+ str(retrieval_emission))      
    