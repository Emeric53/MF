import os
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import pickle
from osgeo import gdal


# 构建查找表
def build_lookup_table(enhancements):
    lookup_table = {}
    basepath = r"C:\\PcModWin5\\Bin\\batch\\AHSI_trans_0_ppmm_tape7.txt"
    ahsi_channels_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
    wavelengths, base_spectrum = get_simulated_satellite_transmittance(basepath,ahsi_channels_path,1000,2500)
    for enhancement in enhancements:
        filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_trans_{int(enhancement)}_ppmm_tape7.txt"
        _, spectrum = get_simulated_satellite_transmittance(filepath,ahsi_channels_path,1000,2500)    
        lookup_table[enhancement] = np.array(spectrum)/np.array(base_spectrum)
    return wavelengths, lookup_table


# 保存查找表到文件 b means binary mode
def save_lookup_table(filename, wavelengths, lookup_table):
    with open(filename, 'wb') as f:
        pickle.dump((wavelengths, lookup_table), f)


# 从文件加载查找表
def load_lookup_table(filename):
    with open(filename, 'rb') as f:
        wavelengths, lookup_table = pickle.load(f)
    return wavelengths, lookup_table


# 插值查找
def lookup_spectrum(enhancement, wavelengths,lookup_table,low_wavelength,high_wavelength):
    condition = np.where((np.array(wavelengths)>= low_wavelength) & (np.array(wavelengths) <= high_wavelength))
    enhancements = np.array(list(lookup_table.keys()))
    spectra = np.array(list(lookup_table.values()))[:,condition]
    interpolator = interp1d(enhancements, spectra, axis=0, fill_value="extrapolate")
    result = np.clip(interpolator(enhancement)[0,:],0,1)
    return np.array(wavelengths)[condition],result


# 加载查找表
def generate_transmittance_cube(plumes: np.ndarray,low_wavelength,high_wavelength): 
    loaded_wavelengths, loaded_lookup_table = load_lookup_table("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_trans_lookup_table_new.pkl")
    used_wavelengths,_ = lookup_spectrum(0, loaded_wavelengths,loaded_lookup_table, low_wavelength,high_wavelength)
    transmittance_cube = np.ones((len(used_wavelengths),plumes.shape[0], plumes.shape[1]))
    for i in range(plumes.shape[1]):
        for j in range(plumes.shape[0]):
            current_concentration = plumes[i,j]
            _,transmittance_cube[:,i,j] = lookup_spectrum(current_concentration,loaded_wavelengths, loaded_lookup_table,low_wavelength,high_wavelength)
            
    return transmittance_cube



# define the gaussian response function
def gaussian_response(wavelengths, center, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    normalization_factor = sigma * np.sqrt(2 * np.pi)
    response = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)/normalization_factor
    return response


# define the convolution function
def convolution(center_wavelengths:list, fwhms:list, raw_wvls:list, raw_data:list):
        # 存储每个波段的卷积结果
    convoluved_data = []
    # 计算每个波段的卷积积分
    for center, fwhm in zip(center_wavelengths, fwhms):
        response = gaussian_response(raw_wvls, center, fwhm)
        product = raw_data * response
        integrated_data = trapz(product, raw_wvls)
        convoluved_data.append(integrated_data)
    return np.array(convoluved_data)


# select the central wavelengths and FWHMs of EMIT that are within the range of 1000-2500 nm
def load_satellite_channels(path,lower_wavelength=1000,upper_wavelength=2500):
    channels = np.load(path)
    central_wavelengths = channels['central_wvls']
    fwhms = channels['fwhms']
    index = np.where((central_wavelengths >= lower_wavelength) & (central_wavelengths <= upper_wavelength))
    used_central_wavelengths = central_wavelengths[index]
    used_fwhms = fwhms[index]
    return used_central_wavelengths, used_fwhms


# read the simulated radiance data from the modtran output file and return the wavelength and radiance
def read_simulated_radiance(path):
    # radiance 卷积
    # 读取 modtran 模拟的光谱数据
    with open(path, 'r') as f:
        radiance_wvl = []
        radiance_list = []
        datalines = f.readlines()[11:-2]
        for data in datalines:
            wvl = 10000000 / float(data[0:9])
            radiance_wvl.append(wvl)
            radiance = float(data[97:108])*10e7/wvl**2*10000
            radiance_list.append(radiance)
    simulated_rad_wavelengths = np.array(radiance_wvl)[::-1]
    simulated_radiance = np.array(radiance_list)[::-1]
    return simulated_rad_wavelengths, simulated_radiance


def read_simulated_transmittance(path):
    with open(path, 'r') as f:
        trans_wvl = []
        trans_list = []
        datalines = f.readlines()[12:-2]
        for data in datalines:
            wvl = 10000000 / float(data[0:9])
            trans_wvl.append(wvl)
            transmittance = float(data[108:115])
            trans_list.append(transmittance)
    simulated_trans_wavelengths = np.array(trans_wvl)[::-1]
    simulated_trans = np.array(trans_list)[::-1]
    return simulated_trans_wavelengths, simulated_trans


def get_simulated_satellite_radiance(radiance_path,channels_path,lower_wavelength,upper_wavelength):
    # read the simulated radiance data
   # read the simulated radiance data
    simulated_rad_wavelengths, simulated_radiance = read_simulated_radiance(radiance_path)
    # load the central wavelengths and FWHMs of the EMIT channels
    central_wavelengths,fwhms = load_satellite_channels(channels_path,lower_wavelength=lower_wavelength,upper_wavelength=upper_wavelength)
    # convolve the simulated radiance with the EMIT response functions
    convoluved_radiance = convolution(central_wavelengths,fwhms,simulated_rad_wavelengths,simulated_radiance)
    return central_wavelengths,convoluved_radiance


def get_simulated_satellite_transmittance(radiance_path,channels_path,lower_wavelength,upper_wavelength):
    # read the simulated radiance data
   # read the simulated radiance data
    simulated_rad_wavelengths, simulated_transmittance = read_simulated_transmittance(radiance_path)
    # load the central wavelengths and FWHMs of the EMIT channels
    central_wavelengths,fwhms = load_satellite_channels(channels_path,lower_wavelength=lower_wavelength,upper_wavelength=upper_wavelength)
    # convolve the simulated radiance with the EMIT response functions
    convoluved_transmittance = convolution(central_wavelengths,fwhms,simulated_rad_wavelengths,simulated_transmittance)
    return central_wavelengths,convoluved_transmittance


def read_tiff(path):
    # 打开 TIFF 文件
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if not dataset:
        print("文件无法打开！")
    else:
        # 获取波段数量
        bands = dataset.RasterCount
        # 初始化一个用于存储所有波段数据的数组
        img_data = []
        # 循环读取每个波段
        for b in range(bands):
            band = dataset.GetRasterBand(b + 1)  # 波段计数从1开始
            data = band.ReadAsArray()  # 将波段数据读取为 NumPy 数组
            img_data.append(data)
        # 将列表转换为 NumPy 数组
        img_array = np.array(img_data)

    # 关闭数据集
    dataset = None
    return img_array


def get_tiff_files(directory):
    tiff_paths = []  # 用于存储完整路径
    tiff_names = []  # 用于存储文件名

    # 遍历指定目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif') or file.endswith('.tiff'):
                full_path = os.path.join(root, file)
                tiff_paths.append(full_path)
                tiff_names.append(file)

    return tiff_paths, tiff_names   

  
def export2tiff(dataarray,outputpath):
    if len(dataarray.shape) == 2:
        rows, cols = dataarray.shape
        bands = 1
    elif len(dataarray.shape) == 3:
        bands, rows, cols = dataarray.shape
    else:
        raise ValueError("dataarray should be a 2D or 3D numpy array")

    # 创建 TIFF 文件
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(outputpath, cols, rows, bands, gdal.GDT_Float64)

    # 如果需要，设置地理变换和投影
    # dataset.SetGeoTransform(geo_transform)  # 设置地理变换（六参数模型）
    # dataset.SetProjection(projection)       # 设置投影

    if bands == 1:
        band = dataset.GetRasterBand(1)
        band.WriteArray(dataarray)
        band.FlushCache()
    else:
        for i in range(bands):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(dataarray[i, :, :])
            band.FlushCache()

    # 清理
    dataset.FlushCache()

    # 关闭文件
    dataset = None

if __name__ == "__main__":
    low_wavelength = 1500
    high_wavelength = 2500
    loaded_wavelengths, loaded_lookup_table = load_lookup_table("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_trans_lookup_table_new.pkl")
    used_wavelengths,_ = lookup_spectrum(0, loaded_wavelengths,loaded_lookup_table, low_wavelength,high_wavelength)
    _,lookup = lookup_spectrum(50000,loaded_wavelengths, loaded_lookup_table,low_wavelength,high_wavelength)
    cube = generate_transmittance_cube(np.array([[61500],]),low_wavelength,high_wavelength)
    import matplotlib.pyplot as plt
    plt.plot(used_wavelengths,cube[:,0,0])
    plt.show()
    