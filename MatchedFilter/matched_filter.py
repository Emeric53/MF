import numpy as np
import sys
import os
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
import MyFunctions.AHSI_data as ad
import MyFunctions.EMIT_data as ed
from MyFunctions.needed_function import open_unit_absorption_spectrum,filter_and_slice,slice_data


# original matched filter algorithm 整幅图像进行计算
def matched_filter(data_cube: np.array, unit_absorption_spectrum: np.array, albedoadjust, iterate, sparsity):        
    """
    Calculate the methane enhancement of the image data based on the original matched filter
    and the unit absorption spectrum.
    
    :param data_array: numpy array of the image data
    :param unit_absorption_spectrum: list of the unit absorption spectrum
    """
    # 获取 以 波段 行数 列数 为顺序的数据
    bands, rows, cols = data_cube.shape
    # 初始化 concentration 数组，大小与卫星数据尺寸一直
    concentration = np.zeros((rows, cols))
    # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
    background_spectrum = np.nanmean(data_cube, axis=(1,2))
    target_spectrum = background_spectrum*unit_absorption_spectrum
    radiancediff_with_bg =data_cube - background_spectrum[:, None,None]
    # 对当前目标光谱的每一行进行去均值操作，得到调整后的光谱，以此为基础计算协方差矩阵，并获得其逆矩阵
    
    d_covariance = data_cube - background_spectrum[:,None,None]
    covariance = np.zeros((bands, bands))
    for row in range(rows): 
        for col in range(cols):
            covariance += np.outer(d_covariance[:, row, col], d_covariance[:, row, col])
    covariance = covariance/(rows*cols)
    covariance_inverse = np.linalg.inv(covariance)
    
    # 判断是否进行反照率校正，若是，则通过背景光谱和实际光谱计算反照率校正因子
    albedo = np.ones((rows, cols))
    if albedoadjust:
        for row in range(rows):
            for col in range(cols):
                albedo[row, col] = (
                        (data_cube[:,row,col].T @ background_spectrum) /
                        (background_spectrum.T @ background_spectrum)
                )
    for row in range(rows):
        for col in range(cols):
            # 基于最优化公式计算每个像素的甲烷浓度增强值
            numerator = (radiancediff_with_bg[:,row,col].T @ covariance_inverse @ target_spectrum)
            denominator = (target_spectrum.T @ covariance_inverse @ target_spectrum)
            concentration[row,col] = numerator/denominator
    
    if iterate:
        for iter_num in range(5):
            print("iteration: No.", iter_num + 1)
            l1filter = np.ones((rows,cols))
            if sparsity:
                for row in rows:
                    for col in cols:
                        l1filter = 1 / (concentration[row,col] + np.finfo(np.float64).tiny)
            
            # 更新背景光谱和目标光谱
            updated_array = data_cube - (albedo*concentration)[None,:,:]*target_spectrum
            # 计算更新后的 背景光谱 和 目标谱
            background_spectrum = np.mean(updated_array, axis=(1,2))
            target_spectrum = np.multiply(background_spectrum, unit_absorption_spectrum)
            
            # 基于新的目标谱 和 背景光谱 计算协方差矩阵
            radiancediff_with_bg = data_cube - background_spectrum[:,None,None]
            d_covariance = data_cube -(albedo*concentration)[None,:,:]*target_spectrum[:,None] - background_spectrum[:,None]
            
            covariance = np.zeros((bands, bands))
            for row in rows:
                for col in cols:
                    covariance += np.outer(d_covariance[:,row,col], d_covariance[:,row,col])
            covariance = covariance/(rows*cols)
            covariance_inverse = np.linalg.inv(covariance)

            # 计算新的甲烷浓度增强值
            for row in rows:
                for col in cols:
                    numerator = (radiancediff_with_bg[:,row,col].T @ covariance_inverse @ target_spectrum) - l1filter[row,col]
                    denominator = albedo * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                    concentration[row,col] = np.maximum(numerator / denominator, 0.0)
    return concentration
 

# modified matched filter algorithm 整幅图像进行计算
def modified_matched_filter(data_cube: np.array, unit_absorption_spectrum: np.array) -> np.array:
    """
    Calculate the methane enhancement of the image data based on the original matched filter
    and the unit absorption spectrum.

    :param data_cube: numpy array of the image data
    :param unit_absorption_spectrum: list of the unit absorption spectrum
    :param is_iterate: flag to decide whether to iterate the matched filter
    :param is_albedo: flag to decide whether to do the albedo correction
    :param is_filter: flag to decide whether to add the l1-filter correction
    :return: numpy array of methane enhancement result
    """
    # 获取 以 波段 行数 列数 为顺序的数据
    bands, rows, cols = data_cube.shape
    # 初始化 concentration 数组，大小与卫星数据尺寸一直
    concentration = np.zeros((rows, cols))
    # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
    background_spectrum = np.nanmean(data_cube, axis=(1,2))
    target_spectrum = background_spectrum*unit_absorption_spectrum
    radiancediff_with_bg = data_cube - background_spectrum[:, None,None]

    d_covariance = data_cube - background_spectrum[:, None,None]
    covariance = np.zeros((bands, bands))
    for row in range(rows): 
        for col in range(cols):
            covariance += np.outer(d_covariance[:, row, col], d_covariance[:, row, col])
    covariance = covariance/(rows*cols)
    covariance_inverse = np.linalg.inv(covariance)

    for row in range(rows):
        for col in range(cols):
            # 基于最优化公式计算每个像素的甲烷浓度增强值
            numerator = (radiancediff_with_bg[:,row,col].T @ covariance_inverse @ target_spectrum)
            denominator = (target_spectrum.T @ covariance_inverse @ target_spectrum)
            concentration[row,col] = numerator/denominator
    mean_concentration = np.nanmean(concentration)
    std_concentration = np.nanstd(concentration)
    adaptive_threshold = mean_concentration + std_concentration
    original_concentration = concentration.copy()
    levelon = True
    while levelon:
        # _, new_unit_absorption_spectrum = lookup_uas_interpolated(adaptive_threshold)
        new_unit_absorption_spectrum = unit_absorption_spectrum.copy()
        high_concentration_mask = concentration > adaptive_threshold
        low_concentration_mask = concentration <= adaptive_threshold
        # 注意：target_spectrum 的更新应该基于更准确的估计值
        background_spectrum = np.nanmean(data_cube - concentration * target_spectrum[:, None, None], axis=(1, 2))
        target_spectrum = background_spectrum * unit_absorption_spectrum
        new_background_spectrum = background_spectrum + adaptive_threshold * new_unit_absorption_spectrum
        high_target_spectrum = new_background_spectrum * new_unit_absorption_spectrum

        radiancediff_with_bg[:,low_concentration_mask] = (
            data_cube[:,low_concentration_mask] - background_spectrum[:,None]
        )
        radiancediff_with_bg[:,high_concentration_mask] = (
            data_cube[:,high_concentration_mask] - new_background_spectrum[:,None]
        )

        d_covariance[:,high_concentration_mask] = data_cube[:,high_concentration_mask] - (
            (concentration[high_concentration_mask]-adaptive_threshold)*high_target_spectrum[:,None] + new_background_spectrum[:,None]
        )
        d_covariance[:,low_concentration_mask] = data_cube[:,low_concentration_mask] - (
            concentration[low_concentration_mask]*target_spectrum[:,None] + background_spectrum[:,None]
        )
        
        covariance = np.zeros((bands, bands))
        for row in range(rows):
            for col in range(cols):
                covariance += np.outer(d_covariance[:, row, col], d_covariance[:, row, col])
        covariance /= rows*cols
        covariance_inverse = np.linalg.inv(covariance)

                
        concentration[high_concentration_mask] = (
            (radiancediff_with_bg[:, high_concentration_mask].T @ covariance_inverse @ high_target_spectrum) / 
            (high_target_spectrum.T @ covariance_inverse @ high_target_spectrum)
        )+ adaptive_threshold

        concentration[low_concentration_mask] =(
            (radiancediff_with_bg[:, low_concentration_mask].T @ covariance_inverse @ target_spectrum) / 
            (target_spectrum.T @ covariance_inverse @ target_spectrum)
        )

        new_mean_concentration = np.nanmean(concentration)
        new_std_concentration = np.nanstd(concentration)
        new_adaptive_threshold = new_mean_concentration + new_std_concentration

        if np.abs((new_adaptive_threshold - adaptive_threshold) / adaptive_threshold) > 0.5:
            adaptive_threshold = new_adaptive_threshold
            print("threshold is unstable, keep on iterating")
        else:
            levelon = False
            print("threshold is stable")
    return concentration
    # mean_concentration = np.nanmean(concentration)
    # std_concentration = np.nanstd(concentration)
    # adaptive_threshold = mean_concentration + std_concentration
    # print("adaptive_threshold:",adaptive_threshold)
    # originalconcentration = concentration.copy()
    # levelon = True
    # while levelon:
    #     _,new_unit_absorption_spectrum = lookup_uas_interpolated(adaptive_threshold)
    #     high_concentration_mask = concentration > adaptive_threshold
    #     low_concentration_mask = concentration <= adaptive_threshold

    #     background_spectrum = np.nanmean(data_cube - concentration * target_spectrum[:, None, None], axis=(1, 2))
    #     target_spectrum = background_spectrum * unit_absorption_spectrum
    #     new_background_spectrum = background_spectrum + adaptive_threshold * unit_absorption_spectrum
    #     high_target_spectrum = new_background_spectrum * new_unit_absorption_spectrum

    #     radiancediff_with_bg[:, high_concentration_mask] = (
    #         data_cube[:, high_concentration_mask]  - (concentration[high_concentration_mask]) *target_spectrum[:, None]
    #         - new_background_spectrum[:, None]
    #     )

    #     radiancediff_with_bg[:, low_concentration_mask] = (
    #         data_cube[:, low_concentration_mask] - concentration[low_concentration_mask] * target_spectrum[:, None]
    #          - background_spectrum[:, None] 
    #     )

    #     covariance = np.zeros((bands, bands))
    #     for row in range(rows):
    #         for col in range(cols):
    #             covariance += np.outer(radiancediff_with_bg[:, row, col], radiancediff_with_bg[:, row, col])
    #     covariance /= rows*cols
    #     covariance_inverse = np.linalg.inv(covariance)

    #     concentration[high_concentration_mask] = (
    #         (radiancediff_with_bg[:, high_concentration_mask].T @ covariance_inverse @ high_target_spectrum) / 
    #         (high_target_spectrum.T @ covariance_inverse @ high_target_spectrum)
    #     ) + adaptive_threshold

    #     concentration[low_concentration_mask] = (
    #         (radiancediff_with_bg[:, low_concentration_mask].T @ covariance_inverse @ target_spectrum) / 
    #         (target_spectrum.T @ covariance_inverse @ target_spectrum)
    #     )
        
    #     new_mean_concentration = np.nanmean(concentration)
    #     new_std_concentration = np.nanstd(concentration)
    #     new_adaptive_threshold = new_mean_concentration + new_std_concentration

    #     if np.abs((new_adaptive_threshold - adaptive_threshold) / adaptive_threshold) < 0.01:
    #         adaptive_threshold = new_adaptive_threshold
    #         print("the distribution of concentration is not stable, keep iterating")
    #     else:
    #         levelon = False
    #         print("the distribution of concentration is stable")
    # return originalconcentration,concentration


# convert the radiance into log space 整幅图像进行计算
def lognormal_matched_filter(data_cube: np.array, unit_absorption_spectrum: np.array):
    # 获取 以 波段 行数 列数 为顺序的数据
    bands, rows, cols = data_cube.shape
    # 初始化 concentration 数组，大小与卫星数据尺寸一直
    concentration = np.zeros((rows, cols))
    # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
    log_background_spectrum = np.nanmean(np.log(data_cube), axis=(1,2))
    background_spectrum = np.exp(log_background_spectrum)
    
    # 对当前目标光谱的每一行进行去均值操作，得到调整后的光谱，以此为基础计算协方差矩阵，并获得其逆矩阵
    radiancediff_with_bg = np.log(data_cube) - log_background_spectrum[:, None,None]
    d_covariance = np.log(data_cube)-background_spectrum[:,None,None]
    covariance = np.zeros((bands, bands))
    for row in range(rows): 
        for col in range(cols):
            covariance += np.outer(d_covariance[:, row, col], d_covariance[:, row, col])
    covariance = covariance/(rows*cols)
    covariance_inverse = np.linalg.inv(covariance)

    for row in range(rows):
        for col in range(cols):
            # 基于最优化公式计算每个像素的甲烷浓度增强值
            numerator = (radiancediff_with_bg[:,row,col].T @ covariance_inverse @ unit_absorption_spectrum)
            denominator = (unit_absorption_spectrum.T @ covariance_inverse @ unit_absorption_spectrum)
            concentration[row,col] = numerator/denominator
    return concentration


# orginal matched filter algorithm 逐列计算
def columnwise_matched_filter(data_cube: np.array, unit_absorption_spectrum: np.array, iterate=False,
                   albedoadjust=False, l1filter=False):
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
    bands, rows, cols = data_cube.shape
    # 初始化 concentration 数组，大小与卫星数据尺寸一直
    concentration = np.zeros((rows, cols))
    for col_index in range(cols):
        # 获取当前列的数据
        current_column = data_cube[:, :, col_index]
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
        radiancediff_with_bg = current_column[:, valid_rows] - background_spectrum[:, None]
        d_covariance = current_column[:, valid_rows] - background_spectrum[:, None]
        covariance = np.zeros((bands, bands))
        for i in range(count_not_nan):
            covariance += np.outer(d_covariance[:, i], d_covariance[:, i])
        covariance = covariance/count_not_nan
        covariance_inverse = np.linalg.inv(covariance)

        # 判断是否进行反照率校正，若是，则通过背景光谱和实际光谱计算反照率校正因子
        albedo = np.ones((rows, cols))
        if albedoadjust:
            albedo[valid_rows, col_index] = (
                    (current_column[:, valid_rows].T @ background_spectrum) /
                    (background_spectrum.T @ background_spectrum)
            )

        # 基于最优化公式计算每个像素的甲烷浓度增强值
        numerator = (radiancediff_with_bg.T @ covariance_inverse @ target_spectrum)
        denominator = albedo[valid_rows, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
        concentration[valid_rows, col_index] = numerator/denominator
        # 判断是否进行迭代，若是，则进行如下迭代计算
        if iterate:
            l1filter = np.zeros((rows, cols))
            epsilon = np.finfo(np.float64).tiny
            for iter_num in range(5):
                print("iteration: No.", iter_num + 1)
                if l1filter:
                    l1filter[valid_rows, col_index] = 1 / (concentration[valid_rows, col_index] + epsilon)
                # 更新背景光谱和目标光谱
                column_replacement = current_column[:, valid_rows] - (albedo[valid_rows, col_index] *concentration[valid_rows, col_index])[None,:]*target_spectrum[:,None]
                # 计算更新后的 背景光谱 和 目标谱
                background_spectrum = np.nanmean(column_replacement, axis=1)
                target_spectrum = np.multiply(background_spectrum, unit_absorption_spectrum)
                radiancediff_with_bg = current_column[:, valid_rows] - background_spectrum
                
                # 基于新的目标谱 和 背景光谱 计算协方差矩阵
                d_covariance = current_column[:, valid_rows] -(albedo[valid_rows, col_index] *concentration[valid_rows, col_index])[None,:]*target_spectrum[:,None] - background_spectrum[:,None]
                covariance = np.zeros((bands, bands))
                for i in range(valid_rows.shape[0]):
                    covariance += np.outer(radiancediff_with_bg[:, i], radiancediff_with_bg[:, i])
                covariance = covariance/count_not_nan
                covariance_inverse = np.linalg.inv(covariance)

                # 计算新的甲烷浓度增强值
                numerator = (radiancediff_with_bg.T @ covariance_inverse @ target_spectrum) - l1filter[valid_rows, col_index]
                denominator = albedo[valid_rows, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                concentration[valid_rows, col_index] = np.maximum(numerator / denominator, 0.0)
    # 返回甲烷浓度增强和反照率校正
    return concentration


# modified matched filter algorithm 逐列计算
def columnwise_modified_matched_filter(data_array: np.array, stacked_unit_absorption_spectrum: np.array, is_iterate=False,
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
            radiancediff_with_bg = current_column[:, valid_rows] - background_spectrum[:, None]
            covariance = np.zeros((bands, bands))
            for i in range(count_not_nan):
                covariance += np.outer(radiancediff_with_bg[:, i], radiancediff_with_bg[:, i])
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
            up = (radiancediff_with_bg.T @ covariance_inverse @ target_spectrum)
            down = albedo[valid_rows, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
            concentration[valid_rows, col_index] = up / down
            
            levelon = True
            # 计算浓度增强值的均值和标准差
            mean_concentration = np.nanmean(concentration[valid_rows, col_index])  # 忽略 NaN 值
            std_concentration = np.nanstd(concentration[valid_rows, col_index])    # 忽略 NaN 值
            # 使用均值加一个标准差作为自适应阈值
            adaptive_threshold = mean_concentration + std_concentration
            while levelon:
                high_concentration_mask = concentration[valid_rows, col_index] > adaptive_threshold
                # 获取这个阈值的单位吸收谱，可以通过插值查找表获得
                # 使用新的单位吸收谱重新计算目标光谱
                background_spectrum = np.nanmean(current_column[:,valid_rows] + albedo[valid_rows,col_index]*concentration[valid_rows,col_index]*target_spectrum[:, np.newaxis], axis=1)
                background_spectrum = background_spectrum + adaptive_threshold*stacked_unit_absorption_spectrum[1,:]
                target_spectrum = np.multiply(background_spectrum, stacked_unit_absorption_spectrum[1,:])
                radiancediff_with_bg = current_column[:, valid_rows] - background_spectrum[:, None] - albedo[valid_rows,col_index]*(concentration[valid_rows,col_index]-adaptive_threshold)*target_spectrum[:, np.newaxis] 
                covariance = np.zeros((bands, bands))
                for i in range(valid_rows.shape[0]):
                    covariance += np.outer(radiancediff_with_bg[:, i], radiancediff_with_bg[:, i])
                covariance = covariance/count_not_nan
                covariance_inverse = np.linalg.inv(covariance)
                # 基于新的目标光谱重新计算高浓度像素的甲烷浓度增强值
                up = (radiancediff_with_bg[:, high_concentration_mask].T @ covariance_inverse @ target_spectrum)
                down = albedo[valid_rows, col_index][high_concentration_mask] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                # 直接更新原数组
                valid_indices = np.where(valid_rows)[0]
                high_concentration_indices = valid_indices[high_concentration_mask]
                concentration[high_concentration_indices, col_index] = up / down + adaptive_threshold
                # 计算浓度增强值的均值和标准差
                mean_concentration = np.nanmean(concentration[valid_rows, col_index])  # 忽略 NaN 值
                std_concentration = np.nanstd(concentration[valid_rows, col_index])    # 忽略 NaN 值
                # 使用均值加一个标准差作为自适应阈值
                new_adaptive_threshold = mean_concentration + std_concentration
                if np.abs((new_adaptive_threshold-adaptive_threshold)/adaptive_threshold) < 0.1:
                    adaptive_threshold = new_adaptive_threshold
                else:
                    levelon = False

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
                    radiancediff_with_bg = current_column[:, valid_rows] -(albedo[valid_rows, col_index] *concentration[valid_rows, col_index])[None,:]*target_spectrum[:,None] - background_spectrum[:,None]
                    covariance = np.zeros((bands, bands))
                    for i in range(valid_rows.shape[0]):
                        covariance += np.outer(radiancediff_with_bg[:, i], radiancediff_with_bg[:, i])
                    covariance = covariance/count_not_nan
                    covariance_inverse = np.linalg.inv(covariance)

                    # 计算新的甲烷浓度增强值
                    up = (radiancediff_with_bg.T @ covariance_inverse @ target_spectrum) - l1filter[valid_rows, col_index]
                    down = albedo[valid_rows, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                    concentration[valid_rows, col_index] = np.maximum(up / down, 0.0)
                    high_concentration_mask = concentration[valid_rows, col_index] > 5000
                    
                    if np.any(high_concentration_mask):
                        # 使用新的单位吸收谱重新计算目标光谱
                        con = concentration[valid_rows, col_index].copy()
                        background_spectrum = np.nanmean(current_column[:,valid_rows] - albedo[valid_rows,col_index]*con*target_spectrum[:, np.newaxis], axis=1)
                        target_spectrum = np.multiply(background_spectrum, stacked_unit_absorption_spectrum)
                        radiancediff_with_bg = current_column[:, valid_rows] -albedo[valid_rows,col_index]*con*target_spectrum[:, np.newaxis] - background_spectrum[:, None]
                        covariance = np.zeros((bands, bands))
                        for i in range(valid_rows.shape[0]):
                            covariance += np.outer(radiancediff_with_bg[:, i], radiancediff_with_bg[:, i])
                        covariance = covariance/count_not_nan
                        covariance_inverse = np.linalg.inv(covariance)
                        # 基于新的目标光谱重新计算高浓度像素的甲烷浓度增强值
                        up = (radiancediff_with_bg[:, high_concentration_mask].T @ covariance_inverse @ target_spectrum)
                        down = albedo[valid_rows, col_index][high_concentration_mask] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                        # 直接更新原数组
                        valid_indices = np.where(valid_rows)[0]
                        high_concentration_indices = valid_indices[high_concentration_mask]
                        concentration[high_concentration_indices, col_index] = up / down + 2500

    if not is_columnwise:
        count_not_nan = np.count_nonzero(~np.isnan(data_array[0, :, :]))
        background_spectrum = np.nanmean(data_array, axis=(1,2))
        target_spectrum = np.multiply(background_spectrum, stacked_unit_absorption_spectrum[0,:])   
        radiancediff_with_bg = data_array - background_spectrum[:, None, None]
        covariance = np.zeros((bands, bands))
        for i in range(rows):
            for j in range(cols):
                covariance = covariance + np.outer(radiancediff_with_bg[:, i, j], radiancediff_with_bg[:, i, j])
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
                up = (radiancediff_with_bg[:,row_index,col_index].T @ covariance_inverse @ target_spectrum)
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
                radiancediff_with_bg = data_array - background_spectrum[:, None, None]
                covariance = np.zeros((bands, bands))
                for i in range(rows):
                    for j in range(cols):
                        covariance += np.outer(radiancediff_with_bg[:, i, j], radiancediff_with_bg[:, i, j])
                covariance = covariance / count_not_nan
                covariance_inverse = np.linalg.inv(covariance)
                
                for row_index in range(rows):
                    for col_index in range(cols):
                        up = (radiancediff_with_bg[:, row_index, col_index].T @ covariance_inverse @ target_spectrum)
                        down = albedo[row_index, col_index] * (target_spectrum.T @ covariance_inverse @ target_spectrum)
                        concentration[row_index, col_index] = np.maximum(up / down, 0)

    # 返回 甲烷浓度增强的结果
    return concentration


# convert the radiance into log space 逐列计算
def columnwise_lognormal_matched_filter(data_cube: np.array, unit_absorption_spectrum: np.array):
    # 获取 以 波段 行数 列数 为顺序的数据
    bands, rows, cols = data_cube.shape
    # 初始化 concentration 数组，大小与卫星数据尺寸一直
    concentration = np.zeros((rows, cols))
    # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
    background_spectrum = np.nanmean(data_cube, axis=(1,2))
    target_spectrum = background_spectrum*unit_absorption_spectrum

    # 对当前目标光谱的每一行进行去均值操作，得到调整后的光谱，以此为基础计算协方差矩阵，并获得其逆矩阵
    radiancediff_with_bg = data_cube - background_spectrum[:, None,None]
    covariance = np.zeros((bands, bands))
    for row in range(rows): 
        for col in range(cols):
            covariance += np.outer(radiancediff_with_bg[:, row, col], radiancediff_with_bg[:, row, col])
    covariance = covariance/(rows*cols)
    covariance_inverse = np.linalg.inv(covariance)

    for row in range(rows):
        for col in range(cols):
            # 基于最优化公式计算每个像素的甲烷浓度增强值
            numerator = (radiancediff_with_bg[:,row,col].T @ covariance_inverse @ target_spectrum)
            denominator = (target_spectrum.T @ covariance_inverse @ target_spectrum)
            concentration[row,col] = numerator/denominator
    return concentration


def uas_lut():
    # 读取单位吸收谱的查找表
    enhance_range = np.arange(0, 15500, 500)
    # 读取单位吸收谱的查找表
    dictionary = {}
    for enhance in enhance_range:
        ahsi_unit_absorption_spectrum_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\uas\\AHSI_UAS_{enhance}.txt"
        bands,uas = open_unit_absorption_spectrum(ahsi_unit_absorption_spectrum_path,2100,2500)
        dictionary[str(enhance)] = [bands,uas]
    np.savez('C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\uas\\uas_dict.npz', **dictionary)
    return dictionary


def lookup_uas_interpolated(value):
    """
    通过插值查找最接近的单位吸收谱。
    
    :param value: 需要查找的浓度值
    :param dictionary: 查找表（浓度值与单位吸收谱的映射）
    :return: 单位吸收谱（如果找到返回插值谱，否则返回None）
    """
    dictionary = np.load(r"C:\Users\RS\VSCode\matchedfiltermethod\MyData\uas\uas_dict.npz")
    keys = sorted(dictionary.keys())
    if value in keys:
        return dictionary[value]
    elif value < float(keys[0]) or value > float(keys[-1]):
        return None  # 超出范围，不进行插值
    else:
        # 找到相邻的浓度值
        for i in range(len(keys) - 1):
            if float(keys[i]) < value < float(keys[i + 1]):
                # 线性插值
                low_key, high_key = keys[i], keys[i + 1]
                low_spectrum, high_spectrum = dictionary[low_key], dictionary[high_key]
                fraction = (value - float(low_key)) / (float(high_key) - float(low_key))
                return (1 - fraction) * low_spectrum + fraction * high_spectrum
        return None


if __name__ == '__main__':
    tif_file_path = r"F:\\AHSI_part1\\GF5B_AHSI_E100.0_N26.4_20231004_011029_L10000400374\\GF5B_AHSI_E100.0_N26.4_20231004_011029_L10000400374_SW.tif"
    bands,radiance = ad.get_calibrated_radiance(tif_file_path,2100,2500)
    # define the path of the unit absorption spectrum file
    ahsi_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_unit_absorption_spectrum.txt"
    bands, uas = open_unit_absorption_spectrum(ahsi_unit_absorption_spectrum_path,2100,2500)
    # # call the main function to process the radiance file 
    from matplotlib import pyplot as plt
    c1 = matched_filter(radiance,uas,False,False,False)
    c2 = modified_matched_filter(radiance,uas)
    # c3 = lognormal_matched_filter(radiance,uas)
    fig, ax = plt.subplots(1,3)
    ax1 = ax[0]
    ax2 = ax[1]
    # ax3 = ax[2]
    
    # 创建网格
    x = np.arange(c1.shape[1])  # 列索引
    y = np.arange(c1.shape[0])  # 行索引
    X, Y = np.meshgrid(x, y)
    contour1 = ax1.contourf(X, Y, c1, 20, cmap='RdGy')
    contour2 = ax2.contourf(X, Y, c2, 20, cmap='RdGy')
    # contour3 = ax3.contourf(X,Y,c3, 20, cmap="RdGy")
    plt.colorbar(contour1,label='Methane Enhancement')
    plt.colorbar(contour2,label='Methane Enhancement')
    # plt.colorbar(contour3,label='Methane Enhancement')
    plt.savefig("mf_compare.png")

 