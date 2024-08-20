import os
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from osgeo import gdal

# 构建查找表
def build_lookup_table(enhancements):
    """build a lookup table for transmittance 

    Args:
        enhancements (np.array): the enhancement range of methane 

    Returns:
        np.array, dictionary: return the wavelengths and lookup_table 
    """
    lookup_table = {}
    basepath = r"C:\\PcModWin5\\Bin\\batch\\AHSI_trans_0_ppmm_tape7.txt"
    ahsi_channels_path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_channels.npz"
    wavelengths, base_spectrum = get_simulated_satellite_transmittance(basepath, ahsi_channels_path, 900, 2500)
    for enhancement in enhancements:
        filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_trans_{int(enhancement)}_ppmm_tape7.txt"
        _, spectrum = get_simulated_satellite_transmittance(filepath, ahsi_channels_path, 900, 2500)    
        lookup_table[enhancement] = np.array(spectrum) / np.array(base_spectrum)
    return wavelengths, lookup_table


# 保存查找表
def save_lookup_table(filename, wavelengths, lookup_table):
    """
    Save the lookup table to a file.

    :param filename: Path to the file where the lookup table will be saved
    :param wavelengths: List or array of wavelengths
    :param lookup_table: Dictionary where keys are enhancements and values are spectra
    """
    np.savez(filename, wavelengths=wavelengths, enhancements=list(lookup_table.keys()), spectra=list(lookup_table.values()))


# 从文件加载查找表
def load_lookup_table(filename):
    """
    Load the lookup table from a file.

    :param filename: Path to the file from which the lookup table will be loaded
    :return: Tuple of wavelengths and the lookup table (dictionary of enhancements and spectra)
    """
    data = np.load(filename)
    wavelengths = data['wavelengths']
    enhancements = data['enhancements']
    spectra = data['spectra']
    lookup_table = {enhancement: spectrum for enhancement, spectrum in zip(enhancements, spectra)}
    return wavelengths, lookup_table


# 插值查找
def lookup_spectrum(enhancement, wavelengths, lookup_table, low_wavelength, high_wavelength):
    """
    Interpolate the spectrum for a given enhancement within a specified wavelength range.

    :param enhancement: The enhancement value for which the spectrum is needed
    :param wavelengths: List or array of wavelengths
    :param lookup_table: Dictionary where keys are enhancements and values are spectra
    :param low_wavelength: Lower bound of the wavelength range
    :param high_wavelength: Upper bound of the wavelength range
    :return: Tuple of filtered wavelengths and interpolated spectrum
    """
    condition = np.where((np.array(wavelengths) >= low_wavelength) & (np.array(wavelengths) <= high_wavelength))
    enhancements = np.array(list(lookup_table.keys()))
    spectra = np.array(list(lookup_table.values()))[:, condition]
    interpolator = interp1d(enhancements, spectra, axis=0, fill_value="extrapolate")
    result = np.clip(interpolator(enhancement)[0, :], 0, 1)
    return np.array(wavelengths)[condition], result


# 基于查找表和浓度值获得透射率cube
def generate_transmittance_cube(plumes: np.ndarray, low_wavelength, high_wavelength):
    """
    Generate a transmittance cube based on the lookup table and concentration values.

    :param plumes: 2D NumPy array of concentration values
    :param low_wavelength: Lower bound of the wavelength range
    :param high_wavelength: Upper bound of the wavelength range
    :return: 3D NumPy array of transmittance values
    """
    loaded_wavelengths, loaded_lookup_table = load_lookup_table("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_trans_lookup_table.npz")
    used_wavelengths, _ = lookup_spectrum(0, loaded_wavelengths, loaded_lookup_table, low_wavelength, high_wavelength)
    transmittance_cube = np.ones((len(used_wavelengths), plumes.shape[0], plumes.shape[1]))
    for i in range(plumes.shape[1]):
        for j in range(plumes.shape[0]):
            current_concentration = plumes[i, j]
            _, transmittance_cube[:, i, j] = lookup_spectrum(current_concentration, loaded_wavelengths, loaded_lookup_table, low_wavelength, high_wavelength)
    return transmittance_cube


# 基于单位吸收谱和浓度值获得透射率cube
def generate_transmittance_cube_fromuas(plumes: np.ndarray, low_wavelength, high_wavelength):
    """
    Generate a transmittance cube based on unit absorption spectrum and concentration values.

    :param plumes: 2D NumPy array of concentration values
    :param low_wavelength: Lower bound of the wavelength range
    :param high_wavelength: Upper bound of the wavelength range
    :return: 3D NumPy array of transmittance values
    """
    _,uas = open_unit_absorption_spectrum(r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_unit_absorption_spectrum.txt",low_wavelength,high_wavelength)
    transmittance_cube = np.ones((len(uas), plumes.shape[0], plumes.shape[1]))
    for i in range(plumes.shape[1]):
        for j in range(plumes.shape[0]):
            current_concentration = plumes[i, j]
            transmittance_cube[:, i, j] = 1 + uas * current_concentration
    return np.clip(transmittance_cube, 0, 1)



# 打开单位吸收谱文件
def open_unit_absorption_spectrum(filepath: str, bot, top):
    """
    Open the unit absorption spectrum file, and convert it to a NumPy array.

    :param filepath: Path to the unit absorption spectrum file
    :return: NumPy array of the unit absorption spectrum
    """
    try:
        with open(filepath, 'r') as file:
            uas_list =  np.array([
                [float(line.split(' ')[0]), float(line.split(' ')[1].strip())]
                for line in file.readlines()
            ])
        indice = np.where((uas_list[:,0] >= bot) & (uas_list[:,0] <= top))    
        return uas_list[indice,0][0], uas_list[indice,1][0]
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


# 从 NumPy 数组中筛选数据并获取切片
def filter_and_slice(arr: np.array, min_val: float, max_val: float):
    """
    根据最大最小值阈值，筛选数组并获取原数组的切片。

    :param arr: 输入的 NumPy 数组
    :param min_val: 最小值阈值
    :param max_val: 最大值阈值
    :return: 筛选后的数组和原数组中的切片
    """
    condition = (arr >= min_val) & (arr <= max_val)
    filtered_arr = arr[condition]
    nonzero_indices = np.nonzero(condition)[0]
    if len(nonzero_indices) == 0:
        return filtered_arr, None
    slice_start = nonzero_indices[0]
    slice_end = nonzero_indices[-1] + 1
    arr_slice = slice(slice_start, slice_end)
    return filtered_arr, arr_slice


# 基于波长最低值和最高值对radiance和uas进行切片
def slice_data(radiance_array, uas, low, high):
    """
    Slice radiance and unit absorption spectrum data based on the lowest and highest wavelengths.

    :param radiance_array: NumPy array of radiance data
    :param uas: NumPy array of unit absorption spectrum data
    :param low: Lower bound of the wavelength range
    :param high: Upper bound of the wavelength range
    :return: Tuple of sliced radiance array and used unit absorption spectrum
    """
    _, slice_uas = filter_and_slice(uas[:, 0], low, high)
    used_radiance = radiance_array[slice_uas, :, :]
    used_uas = uas[slice_uas, 1]
    return used_radiance, used_uas


# define the gaussian response function
def gaussian_response(wavelengths, center, fwhm):
    """
    Define the Gaussian response function.

    :param wavelengths: List or array of wavelengths
    :param center: Center wavelength of the Gaussian function
    :param fwhm: Full width at half maximum of the Gaussian function
    :return: Gaussian response function values
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    normalization_factor = sigma * np.sqrt(2 * np.pi)
    response = np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2) / normalization_factor
    return response


# define the convolution function
def convolution(center_wavelengths: list, fwhms: list, raw_wvls: list, raw_data: list):
    """
    Perform convolution of raw data with Gaussian response functions for specific center wavelengths and FWHMs.

    :param center_wavelengths: List of center wavelengths
    :param fwhms: List of full widths at half maximum (FWHMs)
    :param raw_wvls: List of raw wavelengths
    :param raw_data: List or array of raw data
    :return: Convoluted data as a NumPy array
    """
    convoluved_data = []
    for center, fwhm in zip(center_wavelengths, fwhms):
        response = gaussian_response(raw_wvls, center, fwhm)
        product = raw_data * response
        integrated_data = trapz(product, raw_wvls)
        convoluved_data.append(integrated_data)
    return np.array(convoluved_data)


# 选择在1000-2500nm范围内的EMIT的中心波长和FWHMs
def load_satellite_channels(path, lower_wavelength=1000, upper_wavelength=2500):
    """
    Load the central wavelengths and FWHMs of EMIT channels within the range of 1000-2500 nm.

    :param path: Path to the file containing the satellite channel data
    :param lower_wavelength: Lower bound of the wavelength range (default is 1000)
    :param upper_wavelength: Upper bound of the wavelength range (default is 2500)
    :return: Tuple of central wavelengths and FWHMs within the specified range
    """
    channels = np.load(path)
    central_wavelengths = channels['central_wvls']
    fwhms = channels['fwhms']
    index = np.where((central_wavelengths >= lower_wavelength) & (central_wavelengths <= upper_wavelength))
    used_central_wavelengths = central_wavelengths[index]
    used_fwhms = fwhms[index]
    return used_central_wavelengths, used_fwhms


# 从modtran输出文件读取模拟辐亮度数据，返回波长和辐亮度
def read_simulated_radiance(path):
    """
    Read the simulated radiance data from the modtran output file and return the wavelengths and radiance.

    :param path: Path to the modtran output file
    :return: Tuple of wavelengths and radiance
    """
    with open(path, 'r') as f:
        radiance_wvl = []
        radiance_list = []
        datalines = f.readlines()[11:-2]
        for data in datalines:
            wvl = 10000000 / float(data[0:9])
            radiance_wvl.append(wvl)
            radiance = float(data[97:108]) * 10e7 / wvl ** 2 * 10000
            radiance_list.append(radiance)
    simulated_rad_wavelengths = np.array(radiance_wvl)[::-1]
    simulated_radiance = np.array(radiance_list)[::-1]
    return simulated_rad_wavelengths, simulated_radiance


# 从modtran输出文件读取模拟透射数据，返回波长和透射率
def read_simulated_transmittance(path):
    """
    Read the simulated transmittance data from the modtran output file and return the wavelengths and transmittance.

    :param path: Path to the modtran output file
    :return: Tuple of wavelengths and transmittance
    """
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


# 获取模拟辐射数据并与特定的卫星响应函数进行卷积
def get_simulated_satellite_radiance(radiance_path, channels_path, lower_wavelength, upper_wavelength):
    """
    Get the simulated radiance data and convolve it with the specific satellite response functions.

    :param radiance_path: Path to the simulated radiance data file
    :param channels_path: Path to the file containing the satellite channel data
    :param lower_wavelength: Lower bound of the wavelength range
    :param upper_wavelength: Upper bound of the wavelength range
    :return: Tuple of central wavelengths and convolved radiance
    """
    simulated_rad_wavelengths, simulated_radiance = read_simulated_radiance(radiance_path)
    central_wavelengths, fwhms = load_satellite_channels(channels_path, lower_wavelength=lower_wavelength, upper_wavelength=upper_wavelength)
    convoluved_radiance = convolution(central_wavelengths, fwhms, simulated_rad_wavelengths, simulated_radiance)
    return central_wavelengths, convoluved_radiance


# 获取模拟透射数据并与特定的卫星响应函数进行卷积
def get_simulated_satellite_transmittance(radiance_path, channels_path, lower_wavelength, upper_wavelength):
    """
    Get the simulated transmittance data and convolve it with the specific satellite response functions.

    :param radiance_path: Path to the simulated transmittance data file
    :param channels_path: Path to the file containing the satellite channel data
    :param lower_wavelength: Lower bound of the wavelength range
    :param upper_wavelength: Upper bound of the wavelength range
    :return: Tuple of central wavelengths and convolved transmittance
    """
    simulated_rad_wavelengths, simulated_transmittance = read_simulated_transmittance(radiance_path)
    central_wavelengths, fwhms = load_satellite_channels(channels_path, lower_wavelength=lower_wavelength, upper_wavelength=upper_wavelength)
    convoluved_transmittance = convolution(central_wavelengths, fwhms, simulated_rad_wavelengths, simulated_transmittance)
    return central_wavelengths, convoluved_transmittance



def image_simulation(radiance_path, plume, scaling_factor=1, lower_wavelength=2150, upper_wavelength=2500, row_num=100, col_num=100, noise_level=0.005):
    """
    Simulate a radiance image with added plume effects and Gaussian noise.

    Parameters:
    radiance_path (str): Path to the file containing simulated radiance data.
    plume (array-like): Data representing the plume to be simulated.
    scaling_factor (float, optional): Scaling factor for radiance values. Default is 1.
    lower_wavelength (int, optional): Lower bound of the wavelength range in nm. Default is 2150.
    upper_wavelength (int, optional): Upper bound of the wavelength range in nm. Default is 2500.
    row_num (int, optional): Number of rows in the simulated image. Default is 100.
    col_num (int, optional): Number of columns in the simulated image. Default is 100.
    noise_level (float, optional): Standard deviation of the Gaussian noise relative to the signal. Default is 0.005.

    Returns:
    numpy.ndarray: Simulated radiance image with added plume effects and Gaussian noise.

    The function performs the following steps:
    1. Loads the simulated radiance spectrum from the specified path.
    2. Sets the shape of the image to be simulated.
    3. Generates a universal radiance cube image.
    4. Adds the plume effects to the radiance image.
    5. Adds Gaussian noise to the image.
    """
    # Load the simulated emit radiance spectrum
    channels_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_channels.npz"
    bands, simulated_convolved_spectrum = get_simulated_satellite_radiance(radiance_path, channels_path, lower_wavelength, upper_wavelength)
    
    # Set the shape of the image that want to simulate
    band_num = len(bands)
    
    # Generate the universal radiance cube image
    simulated_image = simulated_convolved_spectrum.reshape(band_num, 1, 1) * np.ones([row_num, col_num])
    
    # Add the Gaussian noise to the image
    cube = generate_transmittance_cube_fromuas(plume*scaling_factor, lower_wavelength, upper_wavelength)
    image_with_plume = cube * simulated_image
    simulated_noisy_image = np.zeros_like(simulated_image)
    for i in range(band_num):  # Traverse each band
        current = simulated_convolved_spectrum[i]
        noise = np.random.normal(0, current * noise_level, (row_num, col_num))  # Generate Gaussian noise
        simulated_noisy_image[i, :, :] = image_with_plume[i, :, :] + noise  # Add noise to the original data
    
    return simulated_noisy_image


# read the tiff file and export the data as a numpy array
def read_tiff(filepath):
    """
    Reads a TIFF file and returns a NumPy array containing all the bands.

    :param filepath: Path to the TIFF file
    :return: a 3D NumPy array with shape (bands, height, width)
    """
    try:
        # 打开文件路径中的数据集
        dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(f"Unable to open file: {filepath}")
    except Exception as ex:
        print(f"Error: {ex}")
        return None

    # 获取波段数
    band_count = dataset.RasterCount
    # 创建一个 NumPy 数组来存储所有波段的数据
    data_array = np.array([dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(band_count)], dtype=np.float32)

    # 关闭数据集
    dataset = None

    return data_array


# get all the tiff files in the directory, return the full path and the file name
def get_tiff_files(directory):
    """
    get all the tiff files in the directory, return the full path and the file name

    :directory: the directory path
    :return: two list containing the full path and the file name
    """
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


# Export a NumPy array to a TIFF file, optionally with the same geo-referencing as a reference file.
def export_to_tiff(dataarray, outputpath, reference_filepath=None):
    """
    Export a NumPy array to a TIFF file, optionally with the same geo-referencing as a reference file.

    :param dataarray: NumPy array to be exported
    :param outputpath: Path to save the output TIFF file
    :param reference_filepath: (Optional) Path to a reference GeoTIFF file for geo-referencing information
    """
    if len(dataarray.shape) == 2:
        rows, cols = dataarray.shape
        bands = 1
    elif len(dataarray.shape) == 3:
        bands, rows, cols = dataarray.shape
    else:
        raise ValueError("dataarray should be a 2D or 3D numpy array")

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(outputpath, cols, rows, bands, gdal.GDT_Float64)

    geo_transform = None
    projection = None

    if reference_filepath:
        ref_dataset = gdal.Open(reference_filepath, gdal.GA_ReadOnly)
        if ref_dataset:
            geo_transform = ref_dataset.GetGeoTransform()
            projection = ref_dataset.GetProjection()

    if geo_transform:
        dataset.SetGeoTransform(geo_transform)
    if projection:
        dataset.SetProjection(projection)

    if bands == 1:
        band = dataset.GetRasterBand(1)
        band.WriteArray(dataarray)
        band.FlushCache()
    else:
        for i in range(bands):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(dataarray[i, :, :])
            band.FlushCache()

    dataset.FlushCache()
    dataset = None

    print(f"File saved successfully at {outputpath}")


if __name__ == "__main__":
    
    ahsi_unit_absorption_spectrum_path = r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_unit_absorption_spectrum.txt"
    # 读取单位吸收谱
    _, uas = open_unit_absorption_spectrum(ahsi_unit_absorption_spectrum_path,2100,2500)
    # path = "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_trans_lookup_table"
    # low_wavelength = 1500
    # high_wavelength = 2500
    # cube = generate_transmittance_cube_fromuas(np.array([[10000],]),900,2500)
    
    # filepath = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_0_ppmm_tape7.txt"
    # channels_path=r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
    # bands,radiance = get_simulated_satellite_radiance(filepath,channels_path,900,2500)
    # filepath2 = f"C:\\PcModWin5\\Bin\\batch\\AHSI_Methane_10000_ppmm_tape7.txt"
    # channels_path=r"C:\\Users\\RS\\VSCode\\matchedfiltermethod\\Needed_data\\AHSI_channels.npz"
    # bands,radiance2 = get_simulated_satellite_radiance(filepath2,channels_path,900,2500) 
    
    # channels,orginal_radiance = read_simulated_radiance(filepath)
    # channels,orginal_radiance2 = read_simulated_radiance(filepath2)
    # # from matplotlib import pyplot as plt
    # # # plt.plot(channels,orginal_radiance)
    # # # plt.plot(channels,orginal_radiance2)
    # # # plt.plot(bands,radiance)
    # # # plt.plot(bands,radiance2)
    # # plt.plot(channels,orginal_radiance2/orginal_radiance)
    # # plt.plot(bands,radiance2/radiance)
    # # plt.ylim(0,1.01)
    # # plt.show()
    
    # from matplotlib import pyplot as plt
    # # 计算比值
    # ratio = radiance * cube[:, 0, 0] / radiance2

    # # 创建主图
    # fig, ax1 = plt.subplots()

    # # 绘制 radiance*cube[:,0,0] 和 radiance2
    # ax1.plot(bands,cube[:, 0, 0], 'r', label='Radiance * Cube')
    # ax1.plot(bands,radiance2/radiance, 'b', label='Radiance2')
    # ax1.set_xlabel('Index')
    # ax1.set_ylabel('Radiance', color='k')
    # ax1.tick_params(axis='y', labelcolor='k')
    # # 添加图例
    # ax1.legend(loc='upper left')

    # # 创建第二个 y 轴
    # ax2 = ax1.twinx()
    # ax2.plot(bands,ratio, 'g', label='Ratio')
    # ax2.set_ylabel('Ratio', color='g')
    # ax2.tick_params(axis='y', labelcolor='g')
    # ax2.set_ylim(0, 1)

    # # 添加图例
    # ax2.legend(loc='upper right')

    # plt.title('Radiance and Ratio Plot')
    # plt.show()
  