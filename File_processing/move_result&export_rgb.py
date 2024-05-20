# This script is used to move the result of the image processing to the corresponding folder and export the RGB image.

import os
import shutil

# import the necessary packages
from osgeo import gdal


def image_coordinate(image_path):
    dataset = gdal.Open(image_path)
    if dataset.GetMetadata('RPC') is None:
        print("RPC file is not found.")
    else:
        corrected_image_path = image_path.replace('tmp.tif', 'rgb.tif')
        warp_options = gdal.WarpOptions(rpc=True)
        corrected_dataset = gdal.Warp(corrected_image_path, dataset, options=warp_options)
        corrected_dataset = None
        dataset = None
        print("校正完成")


def array_to_dataset(array, filepath):
    # 确定数组的行数、列数和波段数
    rows, cols = array.shape[1], array.shape[2]
    bands = 1 if array.ndim == 2 else array.shape[0]

    # 使用内存驱动创建数据集
    driver = gdal.GetDriverByName('GTiff')
    tmp_filepath = filepath.replace('.tif', '_tmp.tif')
    dataset = driver.Create(tmp_filepath, cols, rows, bands, gdal.GDT_Float32)
    rpb_file = filepath.replace('.tif', '.rpb')
    target_rpb = rpb_file.replace('.rpb', '_tmp.rpb')
    shutil.copy(rpb_file, target_rpb)
    # 将 NumPy 数组写入 GDAL 数据集
    for i in range(bands):
        dataset.GetRasterBand(i + 1).WriteArray(array[i, :, :])
    # 关闭 GDAL 数据集
    dataset = None
    image_coordinate(tmp_filepath)

    return None


def get_subdirectories(folder_path):
    """
    获取指定文件夹中所有子文件夹的路径列表。
    :param folder_path: 父文件夹的路径。
    :return: 子文件夹路径列表。
    """
    subdirectories = [os.path.join(folder_path, name) for name in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, name))]
    filename = [name for name in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, name))]
    return subdirectories, filename


folder_path = r"F:\ahsi"
sub, filename = get_subdirectories(folder_path)
total = zip(sub, filename)
for folder, name in total:
    image_path = os.path.join(folder, name + '_VN.tif')
    dataset = gdal.Open(image_path)
    image = dataset.ReadAsArray()
    rgb = image[[59, 38, 20], :, :]
    array_to_dataset(rgb, image_path)
