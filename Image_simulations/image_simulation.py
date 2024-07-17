# add the needed libraries
import numpy as np
import sys
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from tools import needed_function

def image_simulation(radiance_path, plume_path, scaling_factor =1, lower_wavelength= 1500, upper_wavelength= 2500, row_num = 100, col_num = 100, noise_level=0.01):
    # load the simulated emit radiance spectrum
    channels_path=r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
    bands,simulated_convolved_spectrum=needed_function.get_simulated_satellite_radiance(radiance_path,channels_path,lower_wavelength,upper_wavelength)
    
    # set the shape of the image that want to simulate
    band_num = len(bands)

    # generate the universal radiance cube image
    simulated_image = simulated_convolved_spectrum.reshape(band_num, 1, 1) * np.ones([row_num, col_num])
    simulated_noisyimage = np.zeros_like(simulated_image)
    # add the gaussian noise to the image
    for i in range(band_num):  # 遍历每个波段
        current = simulated_convolved_spectrum[i]
        noise = np.random.normal(0, current*noise_level, (row_num, col_num))  # 生成高斯噪声
        simulated_noisyimage[i,:,:] = simulated_image[i,:,:] + noise  # 添加噪声到原始数据

    result = simulated_noisyimage.copy()
    plume = np.load(plume_path)*scaling_factor
    cube = needed_function.generate_transmittance_cube(plume,lower_wavelength,upper_wavelength)
    result = cube*simulated_noisyimage
    return result


if __name__ == "__main__":
    # radiance_path = r"C:\\PcModWin5\\Usr\\AHSI_grassland.fl7"
    # plume_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\simulated_plumes\\gaussianplume_1000_5_1000_200.npy"
    names = ["wetland","urban","grassland","desert"]
    for name in names:
        for stability in ['D','E']:
            for windspeed in [2,4,6,8,10]:
                radiance_path = f"C:\\PcModWin5\\Usr\\AHSI_{name}.fl7" 
                plume_path = f"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\plumes\\gaussianplume_1000_{windspeed}_stability_{stability}.npy"
                sf = 1
                result = image_simulation(radiance_path,plume_path,sf, 1500, 2500, 100, 100, 0.01)
                plume  = np.load(plume_path)
                path = f"I:\\simulated_images_nonoise\\{name}_q_1000_u_{windspeed}_stability_{stability}.tif"
                needed_function.export2tiff(result,path)
                print("simulated images generated")
