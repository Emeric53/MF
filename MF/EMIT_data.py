from osgeo import gdal
import numpy as np
import pathlib as pl
import os


# 数据读取相关
def get_raster_array(filepath):
    """
    Reads a raster file and returns a NumPy array containing all the bands.

    :param filepath: the path of the raster file
    :return: a 3D NumPy array with shape (bands, height, width)
    """
    try:
        # 打开文件路径中的数据集
        dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(f"Unable to open file: {filepath}")
    except Exception as e:
        print(f"Error: {e}")
        return None

    # 获取波段数
    band_count = dataset.RasterCount

    # 创建一个 NumPy 数组来存储所有波段的数据
    data_array = np.array([dataset.GetRasterBand(i).ReadAsArray() for i in range(1, band_count + 1)])

    return data_array


# 数据处理与输出
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
