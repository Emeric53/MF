from osgeo import gdal
import numpy as np
import xml.etree.ElementTree as ET
import pathlib as pl
import os
import shutil

from utils.satellites_data.general_functions import (
    save_ndarray_to_tiff,
    read_tiff_in_numpy,
)
# from general_functions import save_ndarray_to_tiff, read_tiff_in_numpy


# 基于 tiff 读取ahsi的数组
def get_ahsi_array(filepath: str) -> np.array:
    """
    Reads a raster file and returns a NumPy array containing all the bands.
    :param filepath: the path of the raster file
    :return: a 3D NumPy array with shape (bands, height, width)
    """
    return read_tiff_in_numpy(filepath)


# 获取ahsi的波段信息
def read_ahsi_bands():
    """
    get bands list of ahsi
    :param band_file:  filepath containing bands wavelength
    :return: bands list
    """
    # 读取校准文件
    wavelengths = np.load(
        "/home/emeric/Documents/GitHub/MF/data/satellite_channels/AHSI_channels.npz"
    )["central_wvls"]
    return wavelengths


# 对ahsi数据进行光谱校正
def get_radiometric_calibration_coefficients(cal_file: str) -> np.ndarray:
    """
    Perform radiation calibration on the AHSI L1 data using calibration coefficients.

    :param cal_file: calibration filepath
    :return: Calibration coefficients
    """
    try:
        # 读取校准文件
        with open(cal_file, "r") as file:
            lines = file.readlines()

        # 提取校准系数
        coeffs = [tuple(map(float, line.split(","))) for line in lines]
        return np.array(coeffs)

    except FileNotFoundError:
        print(f"Error: The calibration file '{cal_file}' was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


# 对ahsi数据进行光谱校正
def radiance_calibration(dataset: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    # 检查数据集的波段数是否与校准系数的数量匹配
    if len(coeffs) != dataset.shape[0]:
        raise ValueError(
            "The number of calibration coefficients does not match "
            "the number of bands in the dataset."
        )

    # 应用校准系数
    for index, (slope, intercept) in enumerate(coeffs):
        dataset[index, :, :] = slope * (dataset[index, :, :]) + intercept

    return dataset


# 获得光谱校正后的辐射亮度
def get_calibrated_radiance(
    filepath: str, bot: float, top: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform radiation calibration on the AHSI L1 data using calibration coefficients and return the output.

    :param filepath: AHSI filepath
    :return: wavelength array, Calibrated dataset as a 3D NumPy array
    """
    calibration_filepath = os.path.dirname(filepath) + "//GF5B_AHSI_RadCal_SWIR.raw"
    ahsi_array = get_ahsi_array(filepath)
    coeffs = get_radiometric_calibration_coefficients(calibration_filepath)
    bands = read_ahsi_bands()
    indices = np.where((bands >= bot) & (bands <= top))[0]
    calibrated_radiance = radiance_calibration(
        ahsi_array[indices, :, :], coeffs[indices]
    )
    return bands[indices], calibrated_radiance


# 从元数据中获取 SZA 和地面高程
def get_sza_altitude(filepath: str):
    xml_file = filepath.replace("_SW.tif", ".xml")
    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 查找center标签的太阳高度角 (在sunElevationAngle标签下)
    sun_elevation = None
    for elem in root.iter("SolarZenith"):
        sun_elevation = float(elem.text)

    if sun_elevation is not None:
        print(f"中心点太阳高度角: {sun_elevation} 度")
        return sun_elevation, 0
    else:
        print("未找到中心点太阳高度角信息")
        return None


# 将反演结果的数组导出为GeoTIFF文件,并使用与输入文件相同的地理参考
def export_ahsi_array_to_tiff(
    result: np.ndarray,
    filepath: str,
    output_folder: str,
    output_filename: str = None,
    orthorectify=False,
):
    """
    Export a NumPy array to a GeoTIFF file with the same geo-referencing as the input file.

    :param result: NumPy array to be exported
    :param filepath: Path to the input GeoTIFF file
    :param output_folder: Folder to save the output GeoTIFF file
    """
    try:
        # get the file name
        input_filename = pl.Path(filepath).name
        # Use the provided output filename if specified, otherwise use the input file name
        filename = output_filename if output_filename else input_filename
        # generate the whole output_folder_file path
        output_path = os.path.join(output_folder, filename)
        # Export with geo-referencing
        save_ndarray_to_tiff(result, output_path, reference_filepath=filepath)
        if orthorectify:
            image_coordinate(output_path)

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except IOError as io_error:
        print(io_error)
    except Exception as e:
        print(f"An error occurred: {e}")


# 将VNIR数据的rgb波段导出为tiff
def export_rgb_tiff(filepath, outputfolder):
    # 从 VNIR 中 提取 rgb真彩 三波段
    calibration_filepath = os.path.dirname(filepath) + "//GF5B_AHSI_RadCal_VNIR.raw"
    ahsi_array = get_ahsi_array(filepath.replace("_SW.tif", "_VN.tif"))
    vn_rpb = filepath.replace("_SW.tif", "_VN.rpb")
    rgb_rpb = (
        outputfolder + "//" + os.path.basename(filepath).replace("_SW.tif", "_RGB.rpb")
    )
    if os.path.exists(rgb_rpb) is False:
        shutil.copy(vn_rpb, rgb_rpb)
    coeffs = get_radiometric_calibration_coefficients(calibration_filepath)
    calibrated_radiance = radiance_calibration(ahsi_array, coeffs)
    red = calibrated_radiance[59, :, :]  # band 60
    green = calibrated_radiance[38, :, :]  # band 39
    blue = calibrated_radiance[19, :, :]  # band 20
    rgb_radiance = np.stack((red, green, blue), axis=0)
    outputpath = os.path.join(
        outputfolder, os.path.basename(filepath).replace("_SW.tif", "_RGB.tif")
    )
    save_ndarray_to_tiff(rgb_radiance, outputpath, reference_filepath=filepath)
    image_coordinate(outputpath)


# 利用rpc文件对影像进行几何校正
def image_coordinate(image_path: str):
    """
    Use RPC file to get the projection for the TIFF file and apply correction.

    :param image_path: Path to the input TIFF file
    :raises FileNotFoundError: If the input file cannot be opened
    :raises Exception: For other errors that occur during processing
    """
    try:
        dataset = gdal.Open(image_path)
        if dataset is None:
            raise FileNotFoundError(f"Unable to open file: {image_path}")

        if dataset.GetMetadata("RPC") is None:
            print("RPC file is not found.")
        else:
            corrected_image_path = pl.Path(image_path).with_stem(
                f"{pl.Path(image_path).stem}_corrected"
            )
            warp_options = gdal.WarpOptions(rpc=True)
            corrected_dataset = gdal.Warp(
                str(corrected_image_path), dataset, options=warp_options
            )
            if corrected_dataset is None:
                raise RuntimeError("GDAL Warp operation failed.")
            corrected_dataset.FlushCache()
            corrected_dataset = None
            print(
                f"Geolocation Correction completed. Corrected file saved at: {corrected_image_path}"
            )

        dataset = None

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except RuntimeError as rte:
        print(f"Runtime error: {rte}")
    except Exception as e:
        print(f"An error occurred: {e}")


# 以下部分是由于读取 导出的 dat格式的 AHSI数据而添加的函数
# ! 对 AHSI数据读取SZA和地面高程
def get_sza_altitude_from_dat(filepath: str):
    hdr_file = filepath.replace(".dat", ".hdr")
    with open(hdr_file, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=")
                if key.strip() == "solar zenith":
                    sza = float(value.strip())
                    print(f"SZA: {sza}")
    return sza, 0


def get_AHSI_radiances_from_dat(dat_file, low, high):
    radiance_cube = gdal.Open(dat_file)
    radiance = radiance_cube.ReadAsArray()[-180:]
    wvls = read_ahsi_bands()
    indices = np.where((wvls >= low) & (wvls <= high))[0]
    radiance = radiance[indices, :, :]
    return wvls[indices], radiance


def extract_wavelengths_from_hdr(hdr_file):
    wavelengths = []
    inside_wavelength_section = False

    # 打开并逐行读取 .hdr 文件
    with open(hdr_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("wavelength"):
                inside_wavelength_section = True
                continue
            if inside_wavelength_section:
                if line.startswith("{"):
                    # 波长数据的开始
                    continue
                elif line.startswith("}"):
                    # 波长数据的结束
                    break
                else:
                    # 添加波长值
                    wavelengths.extend([float(x) for x in line.split(",")])

    return np.array(wavelengths)


if __name__ == "__main__":
    sample_filepath = r"J:\\AHSI_part4\GF5B_AHSI_E83.9_N43.1_20230929_010957_L10000398404\GF5B_AHSI_E83.9_N43.1_20230929_010957_L10000398404_SW.tif"
    filepath = r"C:\Users\RS\Desktop\Lifei_essay_data\GF5B_AHSI_W103.0_N31.3_20220424_003345_L10000118224\GF5B_AHSI_W103.0_N31.3_20220424_003345_L10000118224_SW.tif"
    radiance_cube = get_ahsi_array(filepath)
    print(radiance_cube.shape)
