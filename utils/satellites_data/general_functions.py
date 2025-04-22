import numpy as np
from osgeo import gdal
from scipy.integrate import trapezoid as trapz
import os


## data I/O functions
# read the tiff file and export the data as a numpy array
def read_tiff_in_numpy(filepath: str) -> np.ndarray:
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
    data_array = np.array(
        [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(band_count)],
        dtype=np.float32,
    )

    # 关闭数据集
    dataset = None

    return data_array


# Export a NumPy array to a TIFF file, optionally with the same geo-referencing as a reference file.
def save_ndarray_to_tiff(
    data_array:np.ndarray, output_path:str, reference_filepath: str = None
):
    """
    Export a NumPy array to a TIFF file, optionally with the same geo-referencing as a reference file.

    :param dataarray: NumPy array to be exported
    :param outputpath: Path to save the output TIFF file
    :param reference_filepath: (Optional) Path to a reference GeoTIFF file for geo-referencing information

    """
    if len(data_array.shape) == 2:
        rows, cols = data_array.shape
        bands = 1
    elif len(data_array.shape) == 3:
        bands, rows, cols = data_array.shape
    else:
        raise ValueError("data_array should be a 2D or 3D numpy array")

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_path, cols, rows, bands, gdal.GDT_Float64)

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
        band.WriteArray(data_array)
        band.FlushCache()
    else:
        for i in range(bands):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(data_array[i, :, :])
            band.FlushCache()

    dataset.FlushCache()
    dataset = None

    print(f"File saved successfully at {output_path}")


## Modtran results processing functions
# define the gaussian response function
def gaussian_response_weights(
    center: float, fwhm: float, coaser_wavelengths: np.ndarray
) -> np.ndarray:
    """
    Define the Gaussian response function.

    :param wavelengths: List or array of wavelengths
    :param center: Center wavelength of the Gaussian function
    :param fwhm: Full width at half maximum of the Gaussian function
    :return: Gaussian response function values
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    normalization_factor = sigma * np.sqrt(2 * np.pi)
    response = (
        np.exp(-0.5 * ((coaser_wavelengths - center) / sigma) ** 2)
        / normalization_factor
    )
    return response


# define the convolution function
def convolute_into_higher_spectral_res(
    center_wavelengths: np.ndarray,
    fwhms: np.ndarray,
    raw_wvls: np.ndarray,
    raw_data: np.ndarray,
) -> np.ndarray:
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
        response_weights = gaussian_response_weights(center, fwhm, raw_wvls)
        product = raw_data * response_weights
        integrated_data = trapz(product, raw_wvls)
        convoluved_data.append(integrated_data)
    return np.array(convoluved_data)


# save the central wavelengths and FWHMs of the channels of a certain satellite to a file
def save_satellite_channels(
    wavelengths: np.ndarray, fwhms: np.ndarray, path: str
) -> None:
    """
    Save the central wavelengths and FWHMs of EMIT channels to a file.

    :param wavelengths: Array of central wavelengths
    :param fwhms: Array of FWHMs
    :param path: Path to save the file
    """
    np.savez(path, central_wvls=wavelengths, fwhms=fwhms)
    print(f"File saved successfully at {path}")
    return None


# load the central wavelengths and FWHMs of the channels of a certain satellite from a file
def load_satellite_channels(
    channel_path: str, lower_wavelength: float = 1000, upper_wavelength: float = 2500
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the central wavelengths and FWHMs of EMIT channels within the range of 1000-2500 nm.

    :param path: Path to the file containing the satellite channel data
    :param lower_wavelength: Lower bound of the wavelength range (default is 1000)
    :param upper_wavelength: Upper bound of the wavelength range (default is 2500)
    :return: Tuple of central wavelengths and FWHMs within the specified range
    """
    channels = np.load(channel_path)
    central_wavelengths = channels["central_wvls"]
    fwhms = channels["fwhms"]

    index = np.where(
        (central_wavelengths >= lower_wavelength)
        & (central_wavelengths <= upper_wavelength)
    )
    used_central_wavelengths = central_wavelengths[index]
    used_fwhms = fwhms[index]
    return used_central_wavelengths, used_fwhms


# 从 NumPy 数组中筛选数据并获取切片
def filter_and_slice(array: np.ndarray, min_val: float, max_val: float):
    """
    根据最大最小值阈值，筛选数组并获取原数组的切片。

    :param arr: 输入的 NumPy 数组
    :param min_val: 最小值阈值
    :param max_val: 最大值阈值
    :return: 筛选后的数组和原数组中的切片
    """
    condition = (array >= min_val) & (array <= max_val)
    filtered_arr = array[condition]
    nonzero_indices = np.nonzero(condition)[0]
    if len(nonzero_indices) == 0:
        return filtered_arr, None
    slice_start = nonzero_indices[0]
    slice_end = nonzero_indices[-1] + 1
    arr_slice = slice(slice_start, slice_end)
    return filtered_arr, arr_slice


# 基于波长最低值和最高值对radiance和uas进行切片
def slice_data(wvls: np.ndarray, dataarray: np.ndarray, low, high):
    """
    Slice radiance and unit absorption spectrum data based on the lowest and highest wavelengths.

    :param radiance_array: NumPy array of radiance data
    :param uas: NumPy array of unit absorption spectrum data
    :param low: Lower bound of the wavelength range
    :param high: Upper bound of the wavelength range
    :return: Tuple of sliced radiance array and used unit absorption spectrum
    """
    used_wvls, slice = filter_and_slice(wvls, low, high)
    if slice is None:
        return None, None
    return used_wvls, dataarray[slice]


# 从modtran输出文件读取模拟辐亮度数据，返回波长和辐亮度
def read_simulated_radiance(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the simulated radiance data from the modtran output file and return the wavelengths and radiance.

    :param path: Path to the modtran output file
    :return: Tuple of wavelengths and radiance
    """
    with open(path, "r") as f:
        radiance_wvl = []
        radiance_list = []
        # read the lines containing the radiance data
        datalines = f.readlines()[11:-2]

        # wavenumber = np.array(datalines[:, :9], dtype=float)
        # radiance = np.array(datalines[:, 97:108], dtype=float)
        # wavelength = 10000000 / wavenumber
        # radiance = radiance * 10e7 / wavelength**2 * 10000

        for data in datalines:
            # convert the wavenumber to wavelength
            wvl = 10000000 / float(data[0:9])
            radiance_wvl.append(wvl)
            # convert the radiance in W/cm^2/sr/cm^-1 to W/m^2/sr/nm
            # radiance = float(data[97:108]) * 10e7 / wvl**2 * 10000
            radiance = float(data[97:108])
            radiance_list.append(radiance)
    # reverse the order of the lists since the original order is according to the wavenumber
    # simulated_rad_wavelengths = wavelength[::-1]
    # simulated_radiance = radiance[::-1]
    simulated_rad_wavelengths = np.array(radiance_wvl)[::-1]
    simulated_radiance = np.array(radiance_list)[::-1]
    return simulated_rad_wavelengths, simulated_radiance


# 从modtran输出文件读取模拟透射数据，返回波长和透射率
def read_simulated_transmittance(path: str):
    """
    Read the simulated transmittance data from the modtran output file and return the wavelengths and transmittance.

    :param path: Path to the modtran output file
    :return: Tuple of wavelengths and transmittance
    """
    with open(path, "r") as f:
        trans_wvl = []
        trans_list = []
        # read the lines containing the radiance data
        datalines = f.readlines()[12:-2]
        for data in datalines:
            # convert the wavenumber to wavelength
            wvl = 10000000 / float(data[0:9])
            trans_wvl.append(wvl)
            # extract the transmittance value
            transmittance = float(data[108:115])
            trans_list.append(transmittance)
    # reverse the order of the lists since the original order is according to the wavenumber
    simulated_trans_wavelengths = np.array(trans_wvl)[::-1]
    simulated_trans = np.array(trans_list)[::-1]
    return simulated_trans_wavelengths, simulated_trans


# 获取模拟辐射数据并与特定的卫星响应函数进行卷积
def get_simulated_satellite_radiance(
    radiance_path: str,
    channels_path: str,
    lower_wavelength: float,
    upper_wavelength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the simulated radiance data and convolve it with the specific satellite response functions.

    :param radiance_path: Path to the simulated radiance data file
    :param channels_path: Path to the file containing the satellite channel data
    :param lower_wavelength: Lower bound of the wavelength range
    :param upper_wavelength: Upper bound of the wavelength range
    :return: Tuple of central wavelengths and convolved radiance
    """
    simulated_wavelengths, simulated_radiance = read_simulated_radiance(radiance_path)
    central_wavelengths, fwhms = load_satellite_channels(
        channels_path,
        lower_wavelength,
        upper_wavelength,
    )
    convoluved_radiance = convolute_into_higher_spectral_res(
        central_wavelengths, fwhms, simulated_wavelengths, simulated_radiance
    )
    return central_wavelengths, convoluved_radiance


# 获取模拟透射数据并与特定的卫星响应函数进行卷积
def get_simulated_satellite_transmittance(
    radiance_path: str,
    channels_path: str,
    lower_wavelength: float,
    upper_wavelength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the simulated transmittance data and convolve it with the specific satellite response functions.

    :param radiance_path: Path to the simulated transmittance data file
    :param channels_path: Path to the file containing the satellite channel data
    :param lower_wavelength: Lower bound of the wavelength range
    :param upper_wavelength: Upper bound of the wavelength range
    :return: Tuple of central wavelengths and convolved transmittance
    """
    simulated_wavelengths, simulated_transmittance = read_simulated_transmittance(
        radiance_path
    )
    central_wavelengths, fwhms = load_satellite_channels(
        channels_path,
        lower_wavelength,
        upper_wavelength,
    )
    convoluved_transmittance = convolute_into_higher_spectral_res(
        central_wavelengths, fwhms, simulated_wavelengths, simulated_transmittance
    )
    return central_wavelengths, convoluved_transmittance


## functions for unit absorption spectrum
# 从文件中打开单位吸收谱并返回指定范围内的数据
def open_unit_absorption_spectrum(
    uas_path: str, bot: float, top: float
) -> tuple[np.ndarray, np.ndarray]:
    """open the unit absorption spectrum file and return the data in the specified range

    Args:
        uaspath (str): the path of the unit absorption spectrum file
        bot (float): the lower bound of the wavelength range
        top (float): the upper bound of the wavelength range

    Returns:
        tuple:
        return numpy arrays of the wavelengths and the unit absorption spectrum
    """
    try:
        with open(uas_path, "r") as file:
            uas_list = np.array(
                [
                    [float(line.split(" ")[0]), float(line.split(" ")[1].strip())]
                    for line in file.readlines()
                ]
            )
        indice = np.where((uas_list[:, 0] >= bot) & (uas_list[:, 0] <= top))
        return uas_list[indice, 0][0], uas_list[indice, 1][0]
    except FileNotFoundError:
        print(f"Error: The file '{uas_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


# get all the tiff files in the directory, return the full path and the file name
def get_tiff_files(directory: str):
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
            if file.endswith(".tif") or file.endswith(".tiff"):
                full_path = os.path.join(root, file)
                tiff_paths.append(full_path)
                tiff_names.append(file)

    return tiff_paths, tiff_names
