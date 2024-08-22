import numpy as np
import sys 
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from MyFunctions import needed_function as nf
from matplotlib import pyplot as plt
from MatchedFilter import matched_filter as mf
from Image_simulations import image_simulation as ims
import seaborn as sns


def image_matched_filter(base_array,data_array: np.array, unit_absorption_spectrum: np.array) :
        bands,rows,cols = data_array.shape
        # 对于非空列，取均值作为背景光谱，再乘以单位吸收光谱，得到目标光谱
        background_spectrum = base_array
        target_spectrum = background_spectrum*unit_absorption_spectrum
        # 对当前目标光谱的每一行进行去均值操作，得到调整后的光谱，以此为基础计算协方差矩阵，并获得其逆矩阵
        radiancediff_with_back = data_array - background_spectrum[:,np.newaxis,np.newaxis]
        covariance = np.zeros((bands,bands))
        concentration = np.zeros((rows,cols))
        for i in range(rows):
            for j in range(cols):
                covariance += np.outer(radiancediff_with_back[:,i,j], radiancediff_with_back[:,i,j])
        covariance /= rows*cols
        covariance_inverse = np.linalg.inv(covariance)
        for i in range(rows):
            for j in range(cols):
                up = (radiancediff_with_back[:,i,j].T @ covariance_inverse @ target_spectrum)
                down = target_spectrum.T @ covariance_inverse @ target_spectrum
                concentration[i,j] = up / down
        return concentration


def profile_matched_filter(base_array, data_array: np.array, unit_absorption_spectrum: np.array) :
    background_spectrum = base_array
    target_spectrum = background_spectrum*unit_absorption_spectrum
    concentration, _, _, _ = np.linalg.lstsq(target_spectrum[:, np.newaxis],(data_array - background_spectrum), rcond=None) 
    if concentration > 5000:
        background_spectrum = base_array + 5000*target_spectrum
        target_spectrum = background_spectrum*unit_absorption_spectrum
        concentration, _, _, _ = np.linalg.lstsq(target_spectrum[:, np.newaxis],(data_array - background_spectrum), rcond=None) 
        concentration += 5000
    return concentration


def radiacne_uas_bands(filepath):
        channels_path=r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_channels.npz"
        bands,radiance = nf.get_simulated_satellite_radiance(filepath,channels_path,2100,2500)
        ahsi_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_unit_absorption_spectrum.txt"
        _,uas = nf.open_unit_absorption_spectrum(ahsi_unit_absorption_spectrum_path,2100,2500)
        # base_radiance, used_uas = nf.slice_data(radiance[:,np.newaxis,np.newaxis], uas, 2150, 2500)
        return radiance,uas,bands


def profilelevel_test1():
    """
    测试不同浓度增强下的甲烷模拟影像,使用通用匹配滤波时的浓度结果分布直方图,看看不同的浓度基准是否会影像结果
    """
    enhancements = np.arange(0,20500,500)
    ahsi_unit_absorption_spectrum_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_unit_absorption_spectrum.txt"
    # ahsi_unit_absorption_spectrum_path2 = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum_2500to7500.txt"
    # ahsi_interval_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\AHSI_unit_absorption_spectrum_from5000.txt"
    _,uas = nf.open_unit_absorption_spectrum(ahsi_unit_absorption_spectrum_path,2100,2500)
    # uas2 = nf.open_unit_absorption_spectrum(ahsi_unit_absorption_spectrum_path2)
    # interval_uas = nf.open_unit_absorption_spectrum(ahsi_interval_unit_absorption_spectrum_path)
    print(uas.shape)
    # initiatate variables
    base = None
    concentration = 0
    total_concentration = 0
    biaslists = []
    
    # 拟合甲烷浓度增强
    for enhancement in enhancements:
        filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_{int(enhancement)}_ppmm_tape7.txt"
        channels_path=r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_channels.npz"
        _,radiance = nf.get_simulated_satellite_radiance(filepath,channels_path,2100,2500)
        
        if base is None:
            base = radiance
            continue
        else:
            concentration = profile_matched_filter(base,radiance, uas)
            total_concentration += concentration
            biaslists.append(((concentration-enhancement)/enhancement))
            print("original concentration is " + str(enhancement))
            print("matched filter result is " + str(concentration))
            print("bias is " + str(float(concentration-enhancement)/enhancement))
    return biaslists
    # # visulization
    # plt.plot(enhancements[1:],biaslists)
    # plt.show()
    # print("total bias is "+ str(float(total_concentration/np.sum(enhancements))))


def imagelevel_test0():
    '''
    测试不同浓度增强下的甲烷浓度增强反演结果的模拟影像的匹配滤波算法直接计算结果，以测试无增强时的浓度反演结果分布
    
    '''
    enhancements = np.arange(0,20500,500)
    col_num = 100
    row_num = 100
    
    basefilepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_0_ppmm_tape7.txt"
    base_radiance,used_uas,band = radiacne_uas_bands(basefilepath)
    for enhancement in enhancements:
        filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_{int(enhancement)}_ppmm_tape7.txt"
        used_radiance,used_uas,band = radiacne_uas_bands(filepath)
        
        simulated_noisyimage = np.zeros((len(band), row_num, col_num))
        for i in range(len(band)):  
            current = used_radiance[i]
            noise = np.random.normal(0, current / 100 , (row_num, col_num))  # 生成高斯噪声
            simulated_noisyimage[i,:,:] = np.ones((row_num,col_num))*current + noise   # 添加噪声到原始数据

        concentration = image_matched_filter(base_radiance,simulated_noisyimage,used_uas)
        # concentration =mf.matched_filter(simulated_noisyimage,used_uas)
        print("enhancement is"+str(enhancement))
        print(np.mean(concentration))
        
        # visulization 
        # plt.subplot(1, 2, 1)
        # plt.title("Histogram of Original Image")
        # plt.hist(concentration.flatten(), bins=50, color='blue', alpha=0.7)
        # plt.axvline(max_value_original, color='red', linestyle='dashed', linewidth=1)
        # plt.text(max_value_original, max(hist_original), f'Mode: {max_value_original:.2f}', color='red', ha='right')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Frequency')
        # plt.show()

 
def imagelevel_test1():
    """  测试随机 2% 像素的统一浓度的甲烷浓度增强的反演结果,并以直方图进行绘制  """
    # radiance_path = r"C:\\PcModWin5\\Usr\\AHSI_grassland.fl7"
    # for type in surface_types:
    #     for stability in ['D','E']:
    #         for windspeed in [2,4,6,8,10]:
    #            radiance_path = f"C:\\PcModWin5\\Usr\\AHSI_{name}.fl7"
    
    enhancements = np.arange(10000,20000,2000)
    surface_types = ["wetland","urban","grassland","desert"]
    
    # for type in surface_types:
    resultlist = []
    resultlist2 = []
    for enhancement in enhancements:
        radiance_path = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_0_ppmm_tape7.txt"
        matrix_size = 100
        num_pixels = 200
        enhance_value = enhancement  # 设定增强的强度值
        plume = np.zeros((matrix_size, matrix_size))
        np.random.seed(42)  # 设置随机种子以保证结果可重复
        # 随机选取2%的像素点进行浓度增强
        indices = np.random.choice(matrix_size * matrix_size, num_pixels, replace=False)
        # 将选取的像素点的浓度值设置为增强值
        np.put(plume, indices, enhance_value)
        # 选取剩余的像素点作为未增强的像素点
        all_indices = np.arange(plume.size)
        unenhanced_indices = np.setdiff1d(all_indices, indices)
        # 将选取的像素点转换为行列索引，分别是增强和未增强的像素点
        enhanced_mask = np.unravel_index(indices, (matrix_size, matrix_size))
        
        unenhanced_mask = np.unravel_index(unenhanced_indices, (matrix_size, matrix_size))
        simulated_image = nf.image_simulation(radiance_path,plume,1, 2100, 2500, 100, 100, 0.005)
        # path = f"I:\\simulated_images_nonoise\\{name}_q_1000_u_{windspeed}_stability_{stability}.tif"
        # needed_function.export_to_tiff(result,path)
        # print("simulated images generated")
        
        # 读取 AHSI 单位吸收谱
        uas_filepath = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_unit_absorption_spectrum.txt"
        _,uas = nf.open_unit_absorption_spectrum(uas_filepath,2100,2500)
        
        original,result = mf.modified_matched_filter(simulated_image, uas)
        enhanced = result[enhanced_mask]
        unenhanced = result[unenhanced_mask]
        resultlist.append(np.mean(enhanced))
        print("改动匹配滤波算法")
        print("当前浓度增强为："+ str(enhancement))
        print("增强像素反演均值：", np.mean(enhanced))
        print("非增强像素反演均值：", np.mean(unenhanced))
        print("偏差：",np.mean(enhanced-enhancement)/enhancement)
        print("\n")
        enhanced = original[enhanced_mask]
        unenhanced = original[unenhanced_mask]
        print("当前浓度增强为："+ str(enhancement))
        print("增强像素反演均值：", np.mean(enhanced))
        print("非增强像素反演均值：", np.mean(unenhanced))
        print("偏差：",np.mean(enhanced-enhancement)/enhancement)
        print("\n")
        # print(np.mean(enhanced1))
        # print(np.mean(enhanced1)/enhancement)
        # print(np.mean(unenhanced1))
        
        # result = mf.matched_filter(simulated_image, uas)
        # enhanced = result[enhanced_mask]
        # unenhanced = result[unenhanced_mask]
        # resultlist2.append(np.mean(enhanced))
        # print("传统匹配滤波算法")
        # print("当前浓度增强为："+ str(enhancement))
        # print("增强像素反演均值：", np.mean(enhanced))
        # print("非增强像素反演均值：", np.mean(unenhanced))
        # print("偏差：",np.mean(enhanced-enhancement)/enhancement)
        # print("\n")
    # from matplotlib import pyplot as plt
    # fig, ax = plt.subplots(1,1)
    # ax.plot(enhancements,resultlist)
    # ax.plot(enhancements,resultlist2)
    # plt.savefig("c.png")
        # np.save(f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Image_simulations\\pixelenhancementresult\\{enhancement}",enhanced)
        # # enhancement2 = matched_filter.matched_filter(simulated_image, used_uas,is_iterate=True, is_albedo= True, is_filter= False,is_columnwise=False)
        # print("highest plume concentration is " + str(np.max(plume)))
            
    # np.savez("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Image_simulations\\pixelenhancementresult\\resultdict.npz", enhancements=enhancements, result=result_dict,non_result = result_dict_non,albedo= albedo)


def imagelevel_test2():
    """ 对叠加了高斯扩散模型的甲烷烟羽模拟影像进行浓度反演 """
    
    names = ["wetland","urban","grassland","desert"]
    
    for name in names:
        #     for stability in ['D','E']:
        #         for windspeed in [2,4,6,8,10]:
        radiance_path = f"C:\\PcModWin5\\Usr\\AHSI_{name}.fl7"
        plume_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\plumes\\gaussianplume_1000_2_stability_D.npy"
        plume = np.load(plume_path)
        sf = 1
        simulated_image = ims.image_simulation(radiance_path,plume,sf, 2150, 2500, 100, 100, 0.01)       
        uas_filepath = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum_0to5000.txt"
        _,uas = nf.open_unit_absorption_spectrum(uas_filepath,2100,2500)
        enhancement,_ = mf.matched_filter(simulated_image, uas)
        print("MF highest concentration is " + str(np.max(enhancement)))
        print("interative MF highest concentration is " + str(np.max(enhancement)))
        print("highest plume concentration is " + str(np.max(plume)))
        
        plume_mask = plume > 100
        result_mask = enhancement > 100
        total_mask = plume_mask*result_mask
        molar_mass_CH4 = 16.04  # 甲烷的摩尔质量，g/mol
        molar_volume_STP = 0.0224  # 摩尔体积，m^3/mol at STP
        emission = np.sum(plume[total_mask])*900*(molar_mass_CH4/molar_volume_STP) * 1e-6
        retrieval_emission = np.sum(enhancement[total_mask])*900*(molar_mass_CH4/molar_volume_STP) * 1e-6
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        # 创建图形和子图
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(enhancement[total_mask].flatten(), bins=50, kde=True, color='green', stat='density', ax=ax)
        # sns.histplot(enhancement2[high_concentration_mask].flatten(), bins=30, kde=True, color='green', stat='density', ax=ax)
        plt.savefig(f"{name}_plume_hist.png")
        print("simulated emission is "+ str(emission))  
        print("retrieved emission is "+ str(retrieval_emission))    
        print("bias is "  + str(retrieval_emission/emission))
        
    """对叠加了高斯扩散模型的甲烷烟羽模拟影像进行浓度反演"""


def imagelevel_test3():
    """对叠加了高斯扩散模型的甲烷烟羽模拟影像,使用不同的匹配滤波算法进行浓度增强反演"""
    # radiance_path = r"C:\\PcModWin5\\Usr\\AHSI_grassland.fl7"
    # plume_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\simulated_plumes\\gaussianplume_1000_5_1000_200.npy"
    # for name in names:
    #     for stability in ['D','E']:
    #         for windspeed in [2,4,6,8,10]:
    names = ["wetland","urban","grassland","desert"]
    for name in names:
        radiance_path = f"C:\\PcModWin5\\Usr\\AHSI_{name}.fl7"
        plume_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\plumes\\gaussianplume_1000_2_stability_D.npy"
        plume = np.load(plume_path)
        sf = 1
        simulated_image = ims.image_simulation(radiance_path,plume,sf, 2100, 2500, 100, 100, 0.01)
        # path = f"I:\\simulated_images_nonoise\\{name}_q_1000_u_{windspeed}_stability_{stability}.tif"
        # needed_function.export_to_tiff(result,path)
        # print("simulated images generated")
        
        uas_filepath = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum_0to5000.txt"
        uas = nf.open_unit_absorption_spectrum(uas_filepath,2100,2500)

        
        enhancement,_ = mf.matched_filter(simulated_image, uas,is_iterate=False, is_albedo= False, is_filter= False,is_columnwise=False)

        print("MF highest concentration is " +str(np.max(enhancement)))

        print("highest plume concentration is " + str(np.max(plume)))
        
        plume_mask = plume > 100
        result_mask = enhancement > 100
        total_mask = plume_mask*result_mask
        molar_mass_CH4 = 16.04  # 甲烷的摩尔质量，g/mol
        molar_volume_STP = 0.0224  # 摩尔体积，m^3/mol at STP
        emission = np.sum(plume[total_mask])*900*(molar_mass_CH4/molar_volume_STP) * 1e-6
        retrieval_emission = np.sum(enhancement[total_mask])*900*(molar_mass_CH4/molar_volume_STP) * 1e-6
        retrieval_emission2 = np.sum(enhancement2[total_mask])*900*(molar_mass_CH4/molar_volume_STP) * 1e-6
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        # 创建图形和子图
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(enhancement[total_mask].flatten(), bins=50, kde=True, color='green', stat='density', ax=ax)
        # sns.histplot(enhancement2[high_concentration_mask].flatten(), bins=30, kde=True, color='green', stat='density', ax=ax)
        plt.savefig(f"{name}_plume_hist.png")
        print("simulated emission is "+ str(emission))  
        print("retrieved emission is "+ str(retrieval_emission))    
        print("bias is "  + str(retrieval_emission/emission))
        print("retrieved iterated emission is "+ str(retrieval_emission2))    
        print("bias is "  + str(retrieval_emission2/emission))
    """对叠加了高斯扩散模型的甲烷烟羽模拟影像进行浓度反演"""

    
    
    return None    


if __name__ == "__main__":
    # biaslist = profilelevel_test1()
    # from matplotlib import pyplot as plt
    # plt.plot(np.arange(500,20500,500),biaslist)
    # plt.show()
    imagelevel_test1()
    # imagelevel_test0()






