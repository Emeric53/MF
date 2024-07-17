from osgeo import gdal
import numpy as np
import os
import sys
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from tools.needed_function import read_tiff,get_tiff_files,export2tiff
from MatchedFilter.matched_filter import open_unit_absorption_spectrum,filter_and_slice,matched_filter
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filelist, filenames = get_tiff_files('I:\\simulated_images_nonoise')
    uas_filepath = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum.txt"
    uas = open_unit_absorption_spectrum(uas_filepath)
    _, used_slice = filter_and_slice(uas[:, 0], 2100, 2500)
    # 获取目标反演窗口范围内的 单位吸收谱和辐亮度数据
    used_uas = uas[used_slice, 1]
    for i,filepath in enumerate(filelist):
        outputpath = os.path.join("I:\\simulated_images_nonoise\\result",filenames[i])
        if not os.path.exists(outputpath):
            data = read_tiff(filepath)[-len(used_uas):,:,:]
            average = np.mean(data,axis=(1,2))
            enhancement = matched_filter(data, used_uas,is_iterate=True, is_albedo= True, is_filter= False,is_columnwise=False)
            if enhancement is not None:
                export2tiff(enhancement,outputpath)
                print(filepath + " has been processed")
            