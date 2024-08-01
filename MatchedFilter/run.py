import numpy as np
import os
import pathlib as pl
import sys
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
import Tools.needed_function as nf
import Tools.AHSI_data as ad
import Tools.EMIT_data as ed
import MatchedFilter.matched_filter as mf


def get_subdirectories(folder_path: str):
    """
    获取指定文件夹中所有子文件夹的路径列表。
    :param  folder_path: 父文件夹的路径。
    :return: 子文件夹路径列表, 子文件夹名称列表
    """
    dir_paths = [os.path.join(folder_path, name) for name in os.listdir(folder_path)
                 if os.path.isdir(os.path.join(folder_path, name))]
    dir_names = [name for name in os.listdir(folder_path)
                 if os.path.isdir(os.path.join(folder_path, name))]
    return dir_paths, dir_names


# 批量处理
def runinbatch(satellite_name: str):
    if satellite_name == 'AHSI':
        # 设置 输出文件夹 路径
        outputfolder = r"I:\AHSI_result"
        # 设置 数据文件夹路径 以及 获取文件夹和文件名称列表
        filefolder_list = ["F:\\AHSI_part1", "H:\\AHSI_part2", "L:\\AHSI_part3", "I:\\AHSI_part4"]
        for filefolder in filefolder_list:
            filelist, namelist = get_subdirectories(filefolder)
            # 遍历每一个数据文件夹 并进行处理
            for index in range(len(filelist)):
                # 获取 SW波段的文件路径
                filepath = os.path.join(filelist[index], namelist[index] + '_SW.tif')
                outputfile = os.path.join(outputfolder, namelist[index] + '_SW.tif')
                # 避免重复计算 进行文件是否存在判断
                if os.path.exists(outputfile):
                    pass
                if not os.path.exists(filepath):
                    print(namelist[index] + ' is not exist')
                else:
                    print(namelist[index] + ' is processing')
                    # 读取radiance数据
                    radiance_cube = ad.get_ahsi_array(filepath)
                    cal_file = os.path.join(filelist[index], "GF5B_AHSI_RadCal_SWIR.raw")
                    radiance = ad.rad_calibration(radiance_cube,cal_file)
                    # define the path of the unit absorption spectrum file
                    ahsi_unit_absorption_spectrum_path = r"C:\Users\RS\VSCode\matchedfiltermethod\Needed_data\AHSI_unit_absorption_spectrum.txt"
                    # 读取单位吸收谱,按照波长范围进行筛选
                    all_uas = nf.open_unit_absorption_spectrum(ahsi_unit_absorption_spectrum_path)
                    _, used_slice = nf.filter_and_slice(all_uas[:, 0], 2100, 2500)
                    used_uas = all_uas[used_slice, 1]
                    
                    # 获取目标反演窗口范围内的 单位吸收谱和辐亮度数据
                    ahsi_channels_path = r"C:\Users\RS\VSCode\matchedfiltermethod\Needed_data\AHSI_channels.npz"
                    ahsibands = np.load(ahsi_channels_path)['central_wvls']
                    _, ahsi_slice = nf.filter_and_slice(ahsibands, 2100, 2500)
                    used_radiance = radiance[ahsi_slice, :, :]
                    # call the main function to process the radiance file
                    enhancement = nf.matched_filter(used_radiance, used_uas, is_iterate=True,
                                                    is_albedo=True, is_filter=True,is_columnwise=True)
                    ad.export_array_to_tiff(enhancement, filepath, outputfolder)

    
    elif satellite_name == 'EMIT':
        # define the path of the radiance folder and get the radiance file list with an img suffix
        radiance_folder = "I:\\EMIT\\rad"
        radiance_path_list = pl.Path(radiance_folder).glob('*.nc')
        # get the output file path and get the existing output file list to avoid the repeat process
        result_folder = pl.Path("I:\\EMIT\\methane_result\\Direct_result")
        output = result_folder.glob('*.nc')
        outputfile = []
        for i in output:
            outputfile.append(str(i.name))
        emit_uas = 'C:\\Users\\RS\\PycharmProjects\\Methane\\unit_absorption_spectrum_emit.txt'
        # 读取单位吸收谱
        all_uas = nf.open_unit_absorption_spectrum(emit_uas)
        # 按照波长范围进行筛选，并获得slice 用于 uas 的筛选
        _, uas_slice = nf.filter_and_slice(all_uas[:, 0], 2100, 2500)
        # 获取目标反演窗口范围内的 单位吸收谱和辐亮度数据
        used_uas = all_uas[uas_slice, 1]

        # 按照波长范围进行筛选，并获得slice 用于 radiance 的筛选
        emit_bands = np.load("emit_bands.npy")
        _, emit_slice = nf.filter_and_slice(emit_bands, 2100, 2500)

        # the input includes the radiance file path, the unit absorption spectrum, the output path and the is_iterate flag
        for radiance_path in radiance_path_list:
            current_filename = str(radiance_path.name)
            if current_filename in outputfile:
                continue
            else:
                print(f"{current_filename} is now being processed")
                try:
                    # 读取radiance数据
                    radiance = ed.get_emit_array(radiance_path)
                    used_radiance = radiance[emit_slice, :, :]
                    enhancement = nf.matched_filter(used_radiance, emit_uas, is_iterate=True, is_albedo=True, is_filter=True)
                    ed.export_array_to_nc(enhancement, radiance_path, "I:\\")
                    print(f"{current_filename} has been processed")
                except Exception as e:
                    print(f"{current_filename} has an error")
                    pass
        # define the path of the radiance file
        radiance_path = r"I:\EMIT\Radiation_data\EMIT_L1B_RAD_001_20220810T065132_2222205_041.nc"
        # 读取radiance数据
        radiance = ed.get_emit_array(radiance_path)
        # define the path of the unit absorption spectrum file
        unit_absorption_spectrum_path = r"C:\Users\RS\PycharmProjects\mf\New_ppm_m_EMIT_unit_absorption_spectrum.txt"
        # 读取单位吸收谱
        all_uas = nf.open_unit_absorption_spectrum(unit_absorption_spectrum_path)
        # 按照波长范围进行筛选，并获得slice 用于 uas 的筛选
        _, uas_slice = nf.filter_and_slice(all_uas[:, 0], 2100, 2500)
        # 获取目标反演窗口范围内的 单位吸收谱和辐亮度数据
        used_uas = all_uas[uas_slice, 1]
        # 按照波长范围进行筛选，并获得slice 用于 radiance 的筛选
        emit_bands = np.load("emit_bands.npy")
        _, emit_slice = nf.filter_and_slice(emit_bands, 2100, 2500)
        used_radiance = radiance[emit_slice, :, :]
        # call the main function to process the radiance file
        enhancement = nf.matched_filter(used_radiance, used_uas, is_iterate=False, is_albedo=False, is_filter=False)
        ed.export_array_to_nc(enhancement, radiance_path, result_folder)
    else:
        print("Invalid satellite name, please select from 'AHSI' or 'EMIT'.")


# 单个AHSI文件处理
def rumfor_AHSI(filepath,outputfolder,mf_type):
    # 设置 输出文件夹 路径
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    
    # 避免重复计算 进行文件是否存在判断
    if os.path.exists(outputfile):
        pass
    else:
        # 读取radiance数据
        radiance = ad.get_calibrated_radiance(filepath)
        
        # 读取单位吸收谱
        ahsi_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum.txt"
        uas = nf.open_unit_absorption_spectrum(ahsi_unit_absorption_spectrum_path)
        ahsi_interval_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\AHSI_unit_absorption_spectrum_interval5000.txt"
        interval_uas = nf.open_unit_absorption_spectrum(ahsi_interval_unit_absorption_spectrum_path)
        
        # 按照波长范围进行筛选，并获得slice 用于 radiance的筛选
        used_radiance, used_uas = nf.slice_data(radiance, uas, 2100, 2500)
        _, used_interval_uas = nf.slice_data(radiance, interval_uas, 2100, 2500)
        
        
        # 运行匹配滤波算法,基于标识符选择不同的算法
        if mf_type == 0:
            # call the main function to process the radiance file
            enhancement = nf.matched_filter(used_radiance, used_uas, is_iterate=False,
                                            is_albedo=False, is_filter=False,is_columnwise=True)
        elif mf_type == 1:
            enhancement = nf.modified_matched_filter(used_radiance, used_uas,used_interval_uas, is_iterate=False,
                                            is_albedo=False, is_filter=False,is_columnwise=True)
        else: 
            print("0 for original mf and 1 for modified mf")
        
        # 将结果导出为tiff文件
        ad.export_array_to_tiff(enhancement, filepath, outputfolder)
    

# 单个EMIT文件处理
def rumfor_EMIT(filepath,outputfolder,mf_type):
    # 设置 输出文件夹 路径
    filename = os.path.basename(filepath)
    outputfile = os.path.join(outputfolder, filename)
    # 避免重复计算 进行文件是否存在判断
    if os.path.exists(outputfile):
        pass
    else:
        # 读取radiance数据
        radiance = ed.get_emit_array(filepath)
        emit_bands = ed.get_emit_bands(filepath)
        # define the path of the unit absorption spectrum file
        EMIT_unit_absorption_spectrum_path = r"C:\Users\RS\VSCode\matchedfiltermethod\Needed_data\EMIT_unit_absorption_spectrum.txt"
        uas = nf.open_unit_absorption_spectrum(EMIT_unit_absorption_spectrum_path)
        EMIT_interval_unit_absorption_spectrum_path = r"C:\Users\RS\VSCode\matchedfiltermethod\Needed_data\AHSI_unit_absorption_spectrum.txt"
        interval_uas = nf.open_unit_absorption_spectrum(EMIT_interval_unit_absorption_spectrum_path)
        
        # 按照波长范围进行筛选，并获得slice 用于 radiance的筛选
        _,slice = nf.filter_and_slice(uas[:,0],2150,2500)
        used_uas = uas[:,1][slice]
        used_interval_uas = interval_uas[:,1][slice]
        _,slice = nf.filter_and_slice(emit_bands,2150,2500)
        used_radiance = radiance[slice,:,:]
        print(used_radiance.shape)
        if mf_type == 0:
            # call the main function to process the radiance file
            enhancement,_ = mf.matched_filter(used_radiance, used_uas, is_iterate=True,
                                            is_albedo=True, is_filter=False,is_columnwise=True)
        elif mf_type == 1:
            enhancement,_ = mf.modified_matched_filter(used_radiance, used_uas,used_interval_uas, is_iterate=False,
                                            is_albedo=False, is_filter=False,is_columnwise=True)
        else: 
            print("0 for original mf and 1 for modified mf")
        ed.export_array_to_nc(enhancement,filepath,outputfolder)



if __name__ == "__main__":
    testpaths = ["I:\\EMIT\\radiance_data\\EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc",
                 "I:\\EMIT\\radiance_data\\EMIT_L1B_RAD_001_20220826T065435_2223805_007.nc"]
    outputfolder = "I:\\EMIT\\runfor1"
    for testpath in testpaths:
        rumfor_EMIT(testpath,outputfolder,0)
