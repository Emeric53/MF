import pathlib as pl
import numpy as np
from osgeo import gdal
import os
from MatchedFilter import matched_filter as mf


# 读取 一个文件夹中的所有子文件夹路径 以及文件夹名称
def get_subdirectories(folder_path):
    """
    获取指定文件夹中所有子文件夹的路径列表。
    :param  folder_path: 父文件夹的路径。
    :return: 子文件夹路径列表, 子文件夹名称列表
    """
    dir_paths = [os.path.join(folder_path, name) for name in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, name))]
    dir_names = [name for name in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, name))]
    return dir_paths, dir_names


def rad_calibration(dataset, cal_file="GF5B_AHSI_RadCal_SWIR.raw"):
    """
    Perform radiation calibration on the AHSI L1 data using calibration coefficients.

    :param dataset: 3D NumPy array of shape (bands, height, width)
    :param cal_file: Path to the calibration file containing coefficients
    :return: Calibrated dataset as a 3D NumPy array
    """
    try:
        # 读取校准文件
        with open(cal_file, "r") as file:
            lines = file.readlines()

        # 提取校准系数
        coeffs = [tuple(map(float, line.split(','))) for line in lines]

        # 检查数据集的波段数是否与校准系数的数量匹配
        if len(coeffs) != dataset.shape[0]:
            raise ValueError("The number of calibration coefficients does not match the number of bands in the dataset.")

        # 应用校准系数
        for index, (slope, intercept) in enumerate(coeffs):
            dataset[index, :, :] = slope * dataset[index, :, :] + intercept

        return dataset

    except FileNotFoundError:
        print(f"Error: The calibration file '{cal_file}' was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def open_unit_absorption_spectrum(filepath, min_wavelength, max_wavelength):
    """
    Open the unit absorption spectrum file, filter it by the spectrum range, and convert it to a NumPy array.

    :param filepath: Path to the unit absorption spectrum file
    :param min_wavelength: Minimum wavelength for filtering
    :param max_wavelength: Maximum wavelength for filtering
    :return: NumPy array of the filtered unit absorption spectrum
    """
    try:
        with open(filepath, 'r') as file:
            # 读取文件并过滤波长范围
            uas_list = [
                float(line.split(' ')[1].strip())
                for line in file.readlines()
                if min_wavelength <= float(line.split(' ')[0]) <= max_wavelength
            ]

        # 转换为 NumPy 数组并返回
        return np.array(uas_list)

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def export_array_to_tiff(result, filepath, output_folder):
    """
    Export a NumPy array to a GeoTIFF file with the same georeferencing as the input file.

    :param result: NumPy array to be exported
    :param filepath: Path to the input GeoTIFF file
    :param output_folder: Folder to save the output GeoTIFF file
    """
    try:
        filename = pl.Path(filepath).name
        output_path = os.path.join(output_folder, filename)

        # 打开输入文件
        dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(f"Unable to open file: {filepath}")

        # 获取地理参考信息
        geo_transform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        # 创建输出文件
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(output_path, result.shape[1], result.shape[0], 1, gdal.GDT_Float32)
        if out_dataset is None:
            raise IOError(f"Unable to create file: {output_path}")

        # 设置空间参考信息
        out_dataset.SetProjection(projection)
        out_dataset.SetGeoTransform(geo_transform)

        # 将 NumPy 数组写入输出文件
        out_dataset.GetRasterBand(1).WriteArray(result)

        # 关闭输出文件
        out_dataset = None

        # 调用自定义函数进行进一步处理（假设函数存在）
        image_coordinate(output_path)

        print(f"File saved successfully at {output_path}")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except IOError as io_error:
        print(io_error)
    except Exception as e:
        print(f"An error occurred: {e}")


# use rpb file to get the projection for the tiff file
def image_coordinate(image_path):
    """
    Use RPC file to get the projection for the TIFF file and apply correction.

    :param image_path: Path to the input TIFF file
    """
    try:
        dataset = gdal.Open(image_path)
        if dataset is None:
            raise FileNotFoundError(f"Unable to open file: {image_path}")

        if dataset.GetMetadata('RPC') is None:
            print("RPC file is not found.")
        else:
            corrected_image_path = pl.Path(image_path).with_suffix('_corrected.tif')
            warp_options = gdal.WarpOptions(rpc=True)
            corrected_dataset = gdal.Warp(str(corrected_image_path), dataset, options=warp_options)
            corrected_dataset = None
            dataset = None
            print("Correction completed.")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"An error occurred: {e}")

if '__main__' == __name__:
    # 设置 数据文件夹路径 以及 获取文件夹和文件名称列表
    filefolder1 = "F:\\AHSI_part1"
    filefolder2 = "F:\\AHSI_part2"
    filefolder3 = "F:\\AHSI_part3"
    filefolder4 = "F:\\AHSI_part4"
    filelist, namelist = get_subdirectories(filefolder1)
    # 设置 输出文件夹 路径
    outputfolder = ''
    # 遍历每一个数据文件夹 并进行处理
    for index in range(len(filelist)):
        # 获取 SW波段的文件路径
        filepath = os.path.join(filelist[index], namelist[index]+'_SW.tif')
        # 创建输出文件路径
        # ？ 和 mf_process 函数中的 output_path 有关吗
        outputfile = os.path.join(outputfolder, namelist[index]+'_SW.tif')
        # 避免重复计算 进行文件是否存在判断
        if os.path.exists(outputfile):
            pass
        else:
            print(namelist[index] + ' is processing')
            try:
                mf.matched_filter(data_array=_, unit_absorption_spectrum=_,is_iterate=False, is_albedo=True, is_filter=True)
                #mf_process(filepath,"unit_absorption_spectrum_ahsi.txt", outputfolder, False)
            except Exception as e:
                print(e)

# # 批量计算
# # define the path of the unit absorption spectrum file and open it
# uas_filepath = 'unit_absorption_spectrum_ahsi.txt'
#
# # based on the code to decide process mode
# process_mode = 1
# if process_mode == 0:
#     # run in batch:
#     # define the path of the radiance folder and get the radiance file list with an img suffix
#     radiance_folder = "I:\\EMIT\\rad"
#     radiance_path_list = pl.Path(radiance_folder).glob('*.nc')
#
#     # get the output file path and get the existing output file list to avoid the repeat process
#     root = pl.Path("I:\\EMIT\\methane_result\\Direct_result")
#     output = root.glob('*.nc')
#     outputfile = []
#     for i in output:
#         outputfile.append(str(i.name))
#
#     # the input includes the radiance file path, the unit absorption spectrum, the output path and the is_iterate flag
#     for radiance_path in radiance_path_list:
#         current_filename = str(radiance_path.name)
#         if current_filename in outputfile:
#             continue
#         else:
#             print(f"{current_filename} is now being processed")
#             try:
#                 mf_process(radiance_path, uas_path=uas_filepath, output_path=root, is_iterate=True, is_albedo=True, is_filter=True)
#                 print(f"{current_filename} has been processed")
#             except Exception as e:
#                 print(f"{current_filename} has an error")
#                 pass
# elif process_mode == 1:
#     # run a single file
#     file_path = pl.Path('')
#     single_result_output = pl.Path('')
#     filename = os.path.basename(file_path)
#     print(f"{file_path} is now being processed.")
#     try:
#         mf_process(file_path, uas_path=uas_filepath, output_path=single_result_output, is_iterate=False, is_albedo=True, is_filter=True)
#         print(f"{filename} has been processed")
#     except Exception as e:
#         print(e)
#         print(f"{filename} has an error")
