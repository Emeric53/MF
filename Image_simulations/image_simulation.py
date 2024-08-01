import numpy as np
import sys
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from Tools import needed_function
from MatchedFilter import matched_filter
import seaborn as sns

def image_simulation(radiance_path, plume, scaling_factor =1, lower_wavelength= 2150, upper_wavelength= 2500, row_num = 100, col_num = 100, noise_level=0.005):
    # load the simulated emit radiance spectrum
    channels_path=r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
    bands,simulated_convolved_spectrum=needed_function.get_simulated_satellite_radiance(radiance_path,channels_path,lower_wavelength,upper_wavelength)
    # set the shape of the image that want to simulate
    band_num = len(bands)
    # generate the universal radiance cube image
    simulated_image = simulated_convolved_spectrum.reshape(band_num, 1, 1) * np.ones([row_num, col_num])
    # add the gaussian noise to the image
    cube = needed_function.generate_transmittance_cube_fromuas(plume,lower_wavelength,upper_wavelength)
    imagewithplume = cube*simulated_image
    simulated_noisyimage = np.zeros_like(simulated_image)
    for i in range(band_num):  # 遍历每个波段
        current = simulated_convolved_spectrum[i]
        noise = np.random.normal(0, current*noise_level, (row_num, col_num))  # 生成高斯噪声
        simulated_noisyimage[i,:,:] = imagewithplume[i,:,:] + noise  # 添加噪声到原始数据
    return simulated_noisyimage

def test1():
    # radiance_path = r"C:\\PcModWin5\\Usr\\AHSI_grassland.fl7"
    # plume_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\simulated_plumes\\gaussianplume_1000_5_1000_200.npy"
    names = ["wetland","urban","grassland","desert"]
    # for name in names:
    #     for stability in ['D','E']:
    #         for windspeed in [2,4,6,8,10]:
    for name in names:
        radiance_path = f"C:\\PcModWin5\\Usr\\AHSI_{name}.fl7"
        plume_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\plumes\\gaussianplume_1000_2_stability_D.npy"
        plume = np.load(plume_path)
        sf = 1
        simulated_image = image_simulation(radiance_path,plume,sf, 2150, 2500, 100, 100, 0.01)
        # path = f"I:\\simulated_images_nonoise\\{name}_q_1000_u_{windspeed}_stability_{stability}.tif"
        # needed_function.export_to_tiff(result,path)
        # print("simulated images generated")
        
        uas_filepath = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum_0to5000.txt"
        uas = needed_function.open_unit_absorption_spectrum(uas_filepath)
        _, used_slice = needed_function.filter_and_slice(uas[:, 0], 2150, 2500)
        # 获取目标反演窗口范围内的 单位吸收谱和辐亮度数据
        used_uas = uas[used_slice, 1]
        
        enhancement,_ = matched_filter.matched_filter(simulated_image, used_uas,is_iterate=False, is_albedo= False, is_filter= False,is_columnwise=False)

        enhancement2,_ = matched_filter.matched_filter(simulated_image, used_uas,is_iterate=True, is_albedo= False, is_filter= False,is_columnwise=False)
        print("MF highest concentration is " +str(np.max(enhancement)))
        print("interative MF highest concentration is " +str(np.max(enhancement2)))
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


def test2():
    # radiance_path = r"C:\\PcModWin5\\Usr\\AHSI_grassland.fl7"
    # plume_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\simulated_plumes\\gaussianplume_1000_5_1000_200.npy"
    names = ["wetland","urban","grassland","desert"]
    # for name in names:
    #     for stability in ['D','E']:
    #         for windspeed in [2,4,6,8,10]:
    enhancements = np.arange(0,10000,10)
    result_dict = []
    result_dict_non = []
    albedo = []
    for name in names:
        for enhancement in enhancements:
            radiance_path = f"C:\\PcModWin5\\Usr\\AHSI_{name}.fl7"
            matrix_size = 100
            num_pixels = 200
            enhance_value = enhancement  # 设定增强的强度值
            plume = np.zeros((matrix_size, matrix_size))
            np.random.seed(42)  # 设置随机种子以保证结果可重复
            indices = np.random.choice(matrix_size * matrix_size, num_pixels, replace=False)
            all_indices = np.arange(plume.size)
            unenhanced_indices = np.setdiff1d(all_indices, indices)
            row_indices, col_indices = np.unravel_index(indices, (matrix_size, matrix_size))
            non_row_indices,non_col_indices = np.unravel_index(unenhanced_indices, (matrix_size, matrix_size))
            np.put(plume, indices, enhance_value)
            sf = 1
            
            simulated_image = image_simulation(radiance_path,plume,sf, 2150, 2500, 100, 100, 0.005)
            # path = f"I:\\simulated_images_nonoise\\{name}_q_1000_u_{windspeed}_stability_{stability}.tif"
            # needed_function.export_to_tiff(result,path)
            # print("simulated images generated")
            uas_filepath = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_unit_absorption_spectrum_0to5000.txt"
            uas = needed_function.open_unit_absorption_spectrum(uas_filepath)
            _, used_slice = needed_function.filter_and_slice(uas[:, 0], 2150, 2500)
            used_uas = uas[used_slice, 1]
            
            result,albedofactor = matched_filter.matched_filter(simulated_image, used_uas,is_iterate=False, is_albedo=True, is_filter=False, is_columnwise=False)
            enhanced = result[row_indices,col_indices].flatten()
            unenhanced = result[non_row_indices,non_col_indices].flatten()
            result_dict.append(enhanced)
            result_dict_non.append(unenhanced)
            albedo.append(albedofactor[row_indices,col_indices].flatten())
            # np.save(f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Image_simulations\\pixelenhancementresult\\{enhancement}",enhanced)
            # # enhancement2 = matched_filter.matched_filter(simulated_image, used_uas,is_iterate=True, is_albedo= True, is_filter= False,is_columnwise=False)
            # print("highest plume concentration is " + str(np.max(plume)))
    np.savez("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Image_simulations\\pixelenhancementresult\\resultdict.npz", enhancements=enhancements, result=result_dict,non_result = result_dict_non,albedo= albedo)


if __name__ == "__main__":
    test1()
    # test2()
    # test = np.load("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Image_simulations\\pixelenhancementresult\\2000.npy")
    # matrix_size = 100
    # plume = np.zeros((matrix_size, matrix_size))
    # np.random.seed(42)  # 设置随机种子以保证结果可重复
    # indices = np.random.choice(matrix_size * matrix_size, 200, replace=False)
    # row_indices, col_indices = np.unravel_index(indices, (matrix_size, matrix_size))
    # print(test)
    # print(np.mean(test))
    # from matplotlib import pyplot as plt
    # plt.hist(test,bins=50,color="green")
    # plt.savefig("80.png")
