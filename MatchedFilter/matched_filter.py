import numpy as np
from osgeo import gdal
import numpy as np
import sys
import os
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
import Tools.AHSI_data as ad
import Tools.EMIT_data as ed
from Tools.needed_function import read_tiff,export_to_tiff,get_tiff_files,open_unit_absorption_spectrum,filter_and_slice,slice_data


def matched_filter(data_array: np.array, unit_absorption_spectrum: np.array, is_iterate=False,
                   is_albedo=False, is_filter=False, is_columnwise=False) -> np.array:
    """
    Calculate the methane enhancement of the image data based on the original matched filter
    and the unit absorption spectrum.

    :param data_array: numpy array of the image data
    :param unit_absorption_spectrum: list of the unit absorption spectrum
    :param is_iterate: flag to decide whether to iterate the matched filter
    :param is_albedo: flag to decide whether to do the albedo correction
    :param is_filter: flag to decide whether to add the l1-filter correction
    :return: numpy array of methane enhancement result
    """
    # 获取 以 波段 行数 列数 为顺序的数据
    bands, rows, cols = data_array.shape
    # 初始化 concentration 数组，大小与卫星数据尺寸一直
    concentration = np.zeros((rows, cols))
    # 遍历不同列数，目的是为了消除 不同传感器之间带来的误差
    if is_columnwise:
        for col_index in range(cols):
            # 获取当前列的数据
            current_column = data_array[:, :, col_index]
            # 获取当前列的非空行的 索引 以及 数目
            valid_rows = ~np.isnan(current_column[0, :])
            count_not_nan = np.count_nonzero(valid_rows)
            # 对于全为空的列，直接将浓度值设为 nan
            if count_not_nan == 0:
                concentration[:, col_index] = np.nan
                continue

            # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
            background_spectrum = np.nanmean(current_column, axis=1)
            target_spectrum = background_spectrum*unit_absorption_spectrum

            # 对当前目标光谱的每一行进行去均值操作，得到调整后的光谱，以此为基础计算协方差矩阵，并获得其逆矩阵
            radiancediff_with_back = current_column[:, valid_rows] - background_spectrum[:, None]
            covariance = np.zeros((bands, bands))
            for i in range(count_not_nan):
                covariance += np.outer(radiancediff_with_back[:, i], radiancediff_with_back[:, i])
            covariance = covariance/count_not_nan
            covariance_inverse = np.linalg.inv(covariance)

            # 判断是否进行反照率校正，若是，则通过背景光谱和实际光谱计算反照率校正因子
            albedo = np.ones((rows, cols))
            if is_albedo:
                albedo[valid_rows, col_index] = (
                        (current_column[:, valid_rows].T @ background_spectrum) /
                        (background_spectrum.T @ background_spectrum)
                )

            # 基于最优化公式计算每个像素的甲烷浓度增强值
            up = (radiancediff_with_back.T @ covariance_inverse @ target_spectrum)
            down = albedo[valid_rows, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
            concentration[valid_rows, col_index] = up / down

            # 判断是否进行迭代，若是，则进行如下迭代计算
            if is_iterate:
                l1filter = np.zeros((rows, cols))
                epsilon = np.finfo(np.float32).tiny
                for iter_num in range(5):
                    if is_filter:
                        l1filter[valid_rows, col_index] = 1 / (concentration[valid_rows, col_index] + epsilon)
                    else:
                        l1filter[valid_rows, col_index] = 0
                    
                    # 更新背景光谱和目标光谱
                    column_replacement = current_column[:, valid_rows] - (albedo[valid_rows, col_index] *concentration[valid_rows, col_index])[None,:]*target_spectrum[:,None]
                    
                    # 计算更新后的 背景光谱 和 目标谱
                    background_spectrum = np.mean(column_replacement, axis=1)
                    target_spectrum = np.multiply(background_spectrum, unit_absorption_spectrum)
                    
                    # 基于新的目标谱 和 背景光谱 计算协方差矩阵
                    radiancediff_with_back = current_column[:, valid_rows] -(albedo[valid_rows, col_index] *concentration[valid_rows, col_index])[None,:]*target_spectrum[:,None] - background_spectrum[:,None]
                    covariance = np.zeros((bands, bands))
                    for i in range(valid_rows.shape[0]):
                        covariance += np.outer(radiancediff_with_back[:, i], radiancediff_with_back[:, i])
                    covariance = covariance/count_not_nan
                    covariance_inverse = np.linalg.inv(covariance)

                    # 计算新的甲烷浓度增强值
                    up = (radiancediff_with_back.T @ covariance_inverse @ target_spectrum) - l1filter[valid_rows, col_index]
                    down = albedo[valid_rows, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                    concentration[valid_rows, col_index] = np.maximum(up / down, 0.0)

    if not is_columnwise:
        count_not_nan = np.count_nonzero(~np.isnan(data_array[0, :, :]))
        background_spectrum = np.nanmean(data_array, axis=(1,2))
        target_spectrum = np.multiply(background_spectrum, unit_absorption_spectrum)   
        radiancediff_with_back = data_array - background_spectrum[:, None, None]
        covariance = np.zeros((bands, bands))
        for i in range(rows):
            for j in range(cols):
                covariance = covariance + np.outer(radiancediff_with_back[:, i, j], radiancediff_with_back[:, i, j])
        covariance = covariance / count_not_nan
        covariance_inverse = np.linalg.inv(covariance)
        albedo = np.ones((rows, cols))
        for row_index in range(rows):
            for col_index in range(cols):
                if is_albedo:
                    albedo[row_index, col_index] = (
                        (data_array[:, row_index, col_index].T @ background_spectrum) /
                        (background_spectrum.T @ background_spectrum)
                    )
                up = (radiancediff_with_back[:,row_index,col_index].T @ covariance_inverse @ target_spectrum)
                down = albedo[row_index, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                concentration[row_index, col_index] = up / down
        
        if is_iterate:
            l1filter = np.zeros((rows, cols))
            epsilon = np.finfo(np.float32).tiny
            
            for iter_num in range(5):
                if is_filter:
                    l1filter = 1 / (concentration + epsilon)
                iter_data = data_array - (
                    target_spectrum[:, None, None] * albedo[None, :, :] * concentration[None, :, :]
                )
                background_spectrum = np.nanmean(iter_data, axis=(1,2))
                target_spectrum = np.multiply(background_spectrum, unit_absorption_spectrum)
                
                radiancediff_with_back = data_array - background_spectrum[:, None, None]
                covariance = np.zeros((bands, bands))
                for i in range(rows):
                    for j in range(cols):
                        covariance += np.outer(radiancediff_with_back[:, i, j], radiancediff_with_back[:, i, j])
                covariance = covariance / count_not_nan
                covariance_inverse = np.linalg.inv(covariance)
                
                for row_index in range(rows):
                    for col_index in range(cols):
                        up = (radiancediff_with_back[:, row_index, col_index].T @ covariance_inverse @ target_spectrum)
                        down = albedo[row_index, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                        concentration[row_index, col_index] = np.maximum(up / down, 0)

    # 返回 甲烷浓度增强的结果
    return concentration,albedo


def modifiedmatched_filter(data_array: np.array, stacked_unit_absorption_spectrum: np.array, is_iterate=False,
                   is_albedo=False, is_filter=False, is_columnwise=False) -> np.array:
    """
    Calculate the methane enhancement of the image data based on the original matched filter
    and the unit absorption spectrum.

    :param data_array: numpy array of the image data
    :param unit_absorption_spectrum: list of the unit absorption spectrum
    :param is_iterate: flag to decide whether to iterate the matched filter
    :param is_albedo: flag to decide whether to do the albedo correction
    :param is_filter: flag to decide whether to add the l1-filter correction
    :return: numpy array of methane enhancement result
    """
    # 获取 以 波段 行数 列数 为顺序的数据
    bands, rows, cols = data_array.shape
    # 初始化 concentration 数组，大小与卫星数据尺寸一直
    concentration = np.zeros((rows, cols))
    # 遍历不同列数，目的是为了消除 不同传感器之间带来的误差
    if is_columnwise:
        for col_index in range(cols):
            # 获取当前列的数据
            current_column = data_array[:, :, col_index]
            # 获取当前列的非空行的 索引 以及 数目
            valid_rows = ~np.isnan(current_column[0, :])
            count_not_nan = np.count_nonzero(valid_rows)
            # 对于全为空的列，直接将浓度值设为 nan
            if count_not_nan == 0:
                concentration[:, col_index] = np.nan
                continue

            # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
            background_spectrum = np.nanmean(current_column, axis=1)
            target_spectrum = background_spectrum*stacked_unit_absorption_spectrum[0,:]

            # 对当前目标光谱的每一行进行去均值操作，得到调整后的光谱，以此为基础计算协方差矩阵，并获得其逆矩阵
            radiancediff_with_back = current_column[:, valid_rows] - background_spectrum[:, None]
            covariance = np.zeros((bands, bands))
            for i in range(count_not_nan):
                covariance += np.outer(radiancediff_with_back[:, i], radiancediff_with_back[:, i])
            covariance = covariance/count_not_nan
            covariance_inverse = np.linalg.inv(covariance)

            # 判断是否进行反照率校正，若是，则通过背景光谱和实际光谱计算反照率校正因子
            albedo = np.ones((rows, cols))
            if is_albedo:
                albedo[valid_rows, col_index] = (
                        (current_column[:, valid_rows].T @ background_spectrum) /
                        (background_spectrum.T @ background_spectrum)
                )

            # 基于最优化公式计算每个像素的甲烷浓度增强值
            up = (radiancediff_with_back.T @ covariance_inverse @ target_spectrum)
            down = albedo[valid_rows, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
            concentration[valid_rows, col_index] = up / down
            
            high_concentration_mask = concentration[valid_rows, col_index] > 5000
            low_concentration_mask = concentration[valid_rows, col_index] <= 5000
            if np.any(high_concentration_mask):
                # 使用新的单位吸收谱重新计算目标光谱
                con = concentration[valid_rows, col_index].copy()
                con[low_concentration_mask] = 2500
                background_spectrum = np.nanmean(current_column[:,valid_rows] - albedo[valid_rows,col_index]*con*target_spectrum[:, np.newaxis], axis=1)
                target_spectrum = np.multiply(background_spectrum, stacked_unit_absorption_spectrum[1,:])
                radiancediff_with_back = current_column[:, valid_rows] -albedo[valid_rows,col_index]*con*target_spectrum[:, np.newaxis] - background_spectrum[:, None]
                covariance = np.zeros((bands, bands))
                for i in range(valid_rows.shape[0]):
                    covariance += np.outer(radiancediff_with_back[:, i], radiancediff_with_back[:, i])
                covariance = covariance/count_not_nan
                covariance_inverse = np.linalg.inv(covariance)
                # 基于新的目标光谱重新计算高浓度像素的甲烷浓度增强值
                up = (radiancediff_with_back[:, high_concentration_mask].T @ covariance_inverse @ target_spectrum)
                down = albedo[valid_rows, col_index][high_concentration_mask] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                # 直接更新原数组
                valid_indices = np.where(valid_rows)[0]
                high_concentration_indices = valid_indices[high_concentration_mask]
                concentration[high_concentration_indices, col_index] = up / down + 2500
            
            # 判断是否进行迭代，若是，则进行如下迭代计算
            if is_iterate:
                l1filter = np.zeros((rows, cols))
                epsilon = np.finfo(np.float32).tiny
                for iter_num in range(5):
                    if is_filter:
                        l1filter[valid_rows, col_index] = 1 / (concentration[valid_rows, col_index] + epsilon)
                    else:
                        l1filter[valid_rows, col_index] = 0
                    
                    # 更新背景光谱和目标光谱
                    column_replacement = current_column[:, valid_rows] - (albedo[valid_rows, col_index] *concentration[valid_rows, col_index])[None,:]*target_spectrum[:,None]
                    # 计算更新后的 背景光谱 和 目标谱
                    background_spectrum = np.mean(column_replacement, axis=1)
                    target_spectrum = np.multiply(background_spectrum, stacked_unit_absorption_spectrum[0,:])
                    # 基于新的目标谱 和 背景光谱 计算协方差矩阵
                    radiancediff_with_back = current_column[:, valid_rows] -(albedo[valid_rows, col_index] *concentration[valid_rows, col_index])[None,:]*target_spectrum[:,None] - background_spectrum[:,None]
                    covariance = np.zeros((bands, bands))
                    for i in range(valid_rows.shape[0]):
                        covariance += np.outer(radiancediff_with_back[:, i], radiancediff_with_back[:, i])
                    covariance = covariance/count_not_nan
                    covariance_inverse = np.linalg.inv(covariance)

                    # 计算新的甲烷浓度增强值
                    up = (radiancediff_with_back.T @ covariance_inverse @ target_spectrum) - l1filter[valid_rows, col_index]
                    down = albedo[valid_rows, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                    concentration[valid_rows, col_index] = np.maximum(up / down, 0.0)
                    high_concentration_mask = concentration[valid_rows, col_index] > 5000
                    
                    if np.any(high_concentration_mask):
                        # 使用新的单位吸收谱重新计算目标光谱
                        con = concentration[valid_rows, col_index].copy()
                        con[low_concentration_mask] = 5000
                        background_spectrum = np.nanmean(current_column[:,valid_rows] - albedo[valid_rows,col_index]*con*target_spectrum[:, np.newaxis], axis=1)
                        target_spectrum = np.multiply(background_spectrum, stacked_unit_absorption_spectrum)
                        radiancediff_with_back = current_column[:, valid_rows] -albedo[valid_rows,col_index]*con*target_spectrum[:, np.newaxis] - background_spectrum[:, None]
                        covariance = np.zeros((bands, bands))
                        for i in range(valid_rows.shape[0]):
                            covariance += np.outer(radiancediff_with_back[:, i], radiancediff_with_back[:, i])
                        covariance = covariance/count_not_nan
                        covariance_inverse = np.linalg.inv(covariance)
                        # 基于新的目标光谱重新计算高浓度像素的甲烷浓度增强值
                        up = (radiancediff_with_back[:, high_concentration_mask].T @ covariance_inverse @ target_spectrum)
                        down = albedo[valid_rows, col_index][high_concentration_mask] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                        # 直接更新原数组
                        valid_indices = np.where(valid_rows)[0]
                        high_concentration_indices = valid_indices[high_concentration_mask]
                        concentration[high_concentration_indices, col_index] = up / down + 2500

    if not is_columnwise:
        count_not_nan = np.count_nonzero(~np.isnan(data_array[0, :, :]))
        background_spectrum = np.nanmean(data_array, axis=(1,2))
        target_spectrum = np.multiply(background_spectrum, stacked_unit_absorption_spectrum[0,:])   
        radiancediff_with_back = data_array - background_spectrum[:, None, None]
        covariance = np.zeros((bands, bands))
        for i in range(rows):
            for j in range(cols):
                covariance = covariance + np.outer(radiancediff_with_back[:, i, j], radiancediff_with_back[:, i, j])
        covariance = covariance / count_not_nan
        covariance_inverse = np.linalg.inv(covariance)
        albedo = np.ones((rows, cols))
        for row_index in range(rows):
            for col_index in range(cols):
                if is_albedo:
                    albedo[row_index, col_index] = (
                        (data_array[:, row_index, col_index].T @ background_spectrum) /
                        (background_spectrum.T @ background_spectrum)
                    )
                up = (radiancediff_with_back[:,row_index,col_index].T @ covariance_inverse @ target_spectrum)
                down = albedo[row_index, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                concentration[row_index, col_index] = up / down
        
        if is_iterate:
            l1filter = np.zeros((rows, cols))
            epsilon = np.finfo(np.float32).tiny
            iter_data = data_array.copy()
            
            for iter_num in range(5):
                if is_filter:
                    l1filter = 1 / (concentration + epsilon)
                iter_data = data_array - (
                    target_spectrum[:, None, None] * albedo[None, :, :] * concentration[None, :, :]
                )
                background_spectrum = np.nanmean(iter_data, axis=(1,2))
                target_spectrum = np.multiply(background_spectrum, stacked_unit_absorption_spectrum[0,:])
                radiancediff_with_back = data_array - background_spectrum[:, None, None]
                covariance = np.zeros((bands, bands))
                for i in range(rows):
                    for j in range(cols):
                        covariance += np.outer(radiancediff_with_back[:, i, j], radiancediff_with_back[:, i, j])
                covariance = covariance / count_not_nan
                covariance_inverse = np.linalg.inv(covariance)
                
                for row_index in range(rows):
                    for col_index in range(cols):
                        up = (radiancediff_with_back[:, row_index, col_index].T @ covariance_inverse @ target_spectrum)
                        down = albedo[row_index, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                        concentration[row_index, col_index] = np.maximum(up / down, 0)

    # 返回 甲烷浓度增强的结果
    return concentration


def get_subfolders(directory):
    subfolders = [os.path.join(directory, folder) for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    return subfolders


if __name__ == '__main__':
    filefolder = get_subfolders(r"F:\\GF5-02_李飞论文所用数据")
    outputfolder = r"F:\\GF5-02_李飞论文所用数据\\result"
    for folder in filefolder:
        folder_name = os.path.basename(folder)
        tif_file_name = f"{folder_name}_sw.tif"
        outputfile = os.path.join(outputfolder, tif_file_name)
        ad.image_coordinate(outputfile)
        tif_file_path = os.path.join(folder, tif_file_name)
        cal_file = os.path.join(folder, "GF5B_AHSI_RadCal_SWIR.raw")
        # 检查文件是否存在
        if os.path.exists(tif_file_path):
            print(f"Processing file: {tif_file_path}")
        else:
            print(f"File not found: {tif_file_path}")
        outputfile = os.path.join(outputfolder, tif_file_name)
            # # 避免重复计算 进行文件是否存在判断
        if os.path.exists(outputfile):
            pass
        else:
            # 读取radiance数据
            radiance = ad.get_ahsi_array(tif_file_path)
            radiance = ad.rad_calibration(radiance,cal_file)
            # define the path of the unit absorption spectrum file
            ahsi_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum.txt"
            ahsi_interval_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum_interval5000.txt"
            # 读取单位吸收谱
            all_uas = open_unit_absorption_spectrum(ahsi_unit_absorption_spectrum_path)
            # 按照波长范围进行筛选，并获得slice 用于 radiance的筛选
            _, used_slice = filter_and_slice(all_uas[:, 0], 2100, 2500)
            # 获取目标反演窗口范围内的 单位吸收谱和辐亮度数据
            used_uas = all_uas[used_slice, 1]
            
            # # 读取单位吸收谱
            # all_interval_uas = open_unit_absorption_spectrum(ahsi_interval_unit_absorption_spectrum_path)
            # # 按照波长范围进行筛选，并获得slice 用于 radiance的筛选
            # _, used_slice = filter_and_slice(all_interval_uas[:, 0], 2100, 2500)
            # # 获取目标反演窗口范围内的 单位吸收谱和辐亮度数据
            # used_interval_uas = all_interval_uas[used_slice, 1]
            
            ahsibands = np.load(r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz")['central_wvls']
            _, ahsi_slice = filter_and_slice(ahsibands, 2100, 2500)
            used_radiance = radiance[ahsi_slice, :, :]
            # if not os.path.exists(MF_outputfile)  :
            #     # call the main function to process the radiance file
            #     mf_enhancement = matched_filter(used_radiance,used_uas,is_iterate = False,is_albedo=True,is_filter=False,is_columnwise=True)
            #     print(mf_enhancement) 
            #     ad.export_array_to_tiff(mf_enhancement, filepath, MF_outputfolder)
            if not os.path.exists(outputfile):
                mf_enhancement = matched_filter(used_radiance, used_uas,is_iterate=True,
                                                is_albedo=True, is_filter=False,is_columnwise=True)
                ad.export_array_to_tiff(mf_enhancement, tif_file_path, outputfolder)
                
    # filepath = r"I:\simulated_images_nonoise\wetland_q_1000_u_6_stability_D.tif"
    # uas_filepath = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum.txt"
    # outputpath = r"I:\simulated_images_nonoise\result\wetland_q_1000_u_6_stability_D.tif"
    
    # uas = open_unit_absorption_spectrum(uas_filepath)
    # _, used_slice = filter_and_slice(uas[:, 0], 2100, 2500)
    # # 获取目标反演窗口范围内的 单位吸收谱和辐亮度数据
    # used_uas = uas[used_slice, 1]
    # data = read_tiff(filepath)[-len(used_uas):,:,:]
    # enhancement = matched_filter(data, used_uas,is_iterate=True, is_albedo= True, is_filter= False,is_columnwise=False)
    # print(enhancement)
    # if enhancement is not None:
    #     export2tiff(enhancement,outputpath)
    #     print(filepath + " has been processed")
    
    
    # # # 设置 输出文件夹 路径
    # MF_outputfolder = r"I:\\mf\\iterate"
    # MMF_outputfolder = r"I:\\mmf\\iterate"
    # filepath = r"F:\\GF5-02_李飞论文所用数据\\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985\\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985_SW.tif"
    # MF_outputfile = os.path.join(MF_outputfolder, 'GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985_SW.tif')
    # MMF_outputfile = os.path.join(MMF_outputfolder, 'GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985_SW.tif')
    # # # 避免重复计算 进行文件是否存在判断
    # if os.path.exists(MF_outputfile) and os.path.exists(MMF_outputfile):
    #     pass
    # else:
    #     # 读取radiance数据
    #     radiance = ad.get_ahsi_array(filepath)
    #     cal_file = os.path.join("F:\GF5-02_李飞论文所用数据\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985", "GF5B_AHSI_RadCal_SWIR.raw")
    #     radiance = ad.rad_calibration(radiance,cal_file)
        
    #     # define the path of the unit absorption spectrum file
    #     ahsi_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum.txt"
    #     ahsi_interval_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum_interval5000.txt"
    #     # 读取单位吸收谱
    #     all_uas = open_unit_absorption_spectrum(ahsi_unit_absorption_spectrum_path)
    #     # 按照波长范围进行筛选，并获得slice 用于 radiance的筛选
    #     _, used_slice = filter_and_slice(all_uas[:, 0], 2100, 2500)
    #     # 获取目标反演窗口范围内的 单位吸收谱和辐亮度数据
    #     used_uas = all_uas[used_slice, 1]
        
    #     # 读取单位吸收谱
    #     all_interval_uas = open_unit_absorption_spectrum(ahsi_interval_unit_absorption_spectrum_path)
    #     # 按照波长范围进行筛选，并获得slice 用于 radiance的筛选
    #     _, used_slice = filter_and_slice(all_interval_uas[:, 0], 2100, 2500)
    #     # 获取目标反演窗口范围内的 单位吸收谱和辐亮度数据
    #     used_interval_uas = all_interval_uas[used_slice, 1]
        
    #     ahsibands = np.load(r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz")['central_wvls']
    #     _, ahsi_slice = filter_and_slice(ahsibands, 2100, 2500)
    #     used_radiance = radiance[ahsi_slice, :, :]
    #     # if not os.path.exists(MF_outputfile)  :
    #     #     # call the main function to process the radiance file
    #     #     mf_enhancement = matched_filter(used_radiance,used_uas,is_iterate = False,is_albedo=True,is_filter=False,is_columnwise=True)
    #     #     print(mf_enhancement) 
    #     #     ad.export_array_to_tiff(mf_enhancement, filepath, MF_outputfolder)
    #     if not os.path.exists(MMF_outputfile):
    #         mmf_enhancement = modifiedmatched_filter(used_radiance, used_uas,used_interval_uas,is_iterate=True,
    #                                         is_albedo=True, is_filter=False,is_columnwise=True)
    #         ad.export_array_to_tiff(mmf_enhancement, filepath, MMF_outputfolder)
   