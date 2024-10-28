import numpy as np
import pathlib as pl

import os
import sys

sys.path.append("c:\\Users\\RS\\VSCode\\matchedfiltermethod\src")
from utils.satellites_data.general_functions import (
    save_ndarray_to_tiff,
    read_tiff_in_numpy,
)


# 读取 enmap 的 radiance 三维数组 要求三维分别是 波段 行 列
def get_enmap_array(file_path: str) -> np.ndarray:
    """
    Reads a raster file and returns a NumPy array containing all the bands.

    :param filepath: the path of the raster file
    :return: a 3D NumPy array with shape (bands, height, width)
    """
    return read_tiff_in_numpy(file_path)


# 从数据文件中 获取 emit的通道波长信息
def read_enmap_bands(file_path: str) -> np.ndarray:
    """
    get bands list of ahsi
    :param band_file:  filepath containing bands wavelength
    :return: bands list
    """
    # 读取校准文件
    wavelengths = np.load(
        "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\satellite_channels\\EnMAP_channels.npz"
    )["central_wvls"]
    return wavelengths


# 获取 筛选范围后的波长数组和radiance信息
def get_enmap_bands_array(
    file_path: str, bot: float, top: float
) -> tuple[np.ndarray, np.ndarray]:
    bands = read_enmap_bands()
    data = get_enmap_array(file_path=file_path)
    indices = np.where((bands >= bot) & (bands <= top))[0]
    return bands[indices], data[indices, :, :]


# 将结果导出为nc文件
def export_enmap_array_to_tiff(
    result: np.ndarray, filepath: str, output_folder: str, output_filename: str = None
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

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except IOError as io_error:
        print(io_error)
    except Exception as e:
        print(f"An error occurred: {e}")


# if __name__ == "__main__":
#     # main()
