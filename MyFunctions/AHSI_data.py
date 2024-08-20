from osgeo import gdal
import numpy as np
import pathlib as pl
import os
from matplotlib import pyplot as plt
import sys 
sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from MyFunctions.needed_function import export_to_tiff,read_tiff

# 数据读取相关
def get_ahsi_array(filepath:str) -> np.array:
    """
    Reads a raster file and returns a NumPy array containing all the bands.

    :param filepath: the path of the raster file
    :return: a 3D NumPy array with shape (bands, height, width)
    """
    return read_tiff(filepath)


# 对ahsi数据进行光谱校正
def get_calibration(cal_file: str ) -> list:
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
        coeffs = [tuple(map(float, line.split(','))) for line in lines]
        return np.array(coeffs)

    except FileNotFoundError:
        print(f"Error: The calibration file '{cal_file}' was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_bands():
    """
    get bands list of ahsi
    :param band_file:  filepath containing bands wavelength
    :return: bands list
    """
    # 读取校准文件
    wavelengths = np.load("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\MyData\\AHSI_channels.npz")["central_wvls"]
    return wavelengths


def rad_calibration(dataset,coeffs):
    # 检查数据集的波段数是否与校准系数的数量匹配
    if len(coeffs) != dataset.shape[0]:
        raise ValueError("The number of calibration coefficients does not match "
                            "the number of bands in the dataset.")

    # 应用校准系数
    for index, (slope, intercept) in enumerate(coeffs):
        dataset[index, :, :] = slope * (dataset[index, :, :]) + intercept
    
    return dataset


# 获得光谱校正后的辐射亮度
def get_calibrated_radiance(filepath, bot, top) -> np.array:
    """
    Perform radiation calibration on the AHSI L1 data using calibration coefficients and return the output.

    :param filepath: AHSI filepath
    :return: Calibrated dataset as a 3D NumPy array
    """
    calibration_filepath = os.path.dirname(filepath) + "//GF5B_AHSI_RadCal_SWIR.raw"
    ahsi_array = get_ahsi_array(filepath)
    coeffs = get_calibration(calibration_filepath)
    bands = get_bands()
    indices = np.where((bands >= bot) & (bands <= top))[0]
    calibrated_radiance = rad_calibration(ahsi_array[indices,:,:], coeffs[indices])
    return bands[indices],calibrated_radiance
    

# 将反演结果的数组导出为GeoTIFF文件,并使用与输入文件相同的地理参考
def export_array_to_tiff(result: np.array, filepath: str, output_folder: str):
    """
    Export a NumPy array to a GeoTIFF file with the same geo-referencing as the input file.

    :param result: NumPy array to be exported
    :param filepath: Path to the input GeoTIFF file
    :param output_folder: Folder to save the output GeoTIFF file
    """
    try:
        # get the file name
        filename = pl.Path(filepath).name

        # generate the whole output_folder_file path
        output_path = os.path.join(output_folder, filename)

        # Export with geo-referencing
        export_to_tiff(result, output_path, reference_filepath=filepath)

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except IOError as io_error:
        print(io_error)
    except Exception as e:
        print(f"An error occurred: {e}")


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

        if dataset.GetMetadata('RPC') is None:
            print("RPC file is not found.")
        else:
            corrected_image_path = pl.Path(image_path).with_stem(f"{pl.Path(image_path).stem}_corrected")
            warp_options = gdal.WarpOptions(rpc=True)
            corrected_dataset = gdal.Warp(str(corrected_image_path), dataset, options=warp_options)
            if corrected_dataset is None:
                raise RuntimeError("GDAL Warp operation failed.")
            corrected_dataset.FlushCache()
            corrected_dataset = None
            print(f"Correction completed. Corrected file saved at: {corrected_image_path}")

        dataset = None

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except RuntimeError as rte:
        print(f"Runtime error: {rte}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    ahsi_file = ("F:\\GF5-02_李飞论文所用数据\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\"
                 "GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222_SW.tif")
    calibrated_radiance = get_calibrated_radiance(ahsi_file)
    plt.plot(calibrated_radiance[:,0,0])
    plt.show()

if __name__ == '__main__':
    a = r"I:\AHSI_part4\GF5B_AHSI_E83.9_N43.1_20230929_010957_L10000398404\GF5B_AHSI_E83.9_N43.1_20230929_010957_L10000398404_SW.tif"
    bands,radiance = get_calibrated_radiance(a, 1500,2300)

