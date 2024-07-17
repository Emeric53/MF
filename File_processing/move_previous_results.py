import os
import shutil
from osgeo import gdal
# This script is used to move the result of the image processing to the corresponding folder 
# and export the RGB image.


def move_previous_result():
    target_path = "I:\\AHSI_result"
    filefolder_list = ["F:\\AHSI_part1", "H:\\AHSI_part2", "L:\\AHSI_part3", "I:\\AHSI_part4"]
    for filefolder in filefolder_list:
        filelist, namelist = get_subdirectories(filefolder)
        # 遍历每一个数据文件夹 并进行处理
        for index in range(len(filelist)):
            # 获取 SW波段的文件路径
            filepath = os.path.join(filelist[index] + '\\result\\' + namelist[index] + '_SW.tif')
            rpbpath = os.path.join(filelist[index] + '\\result\\' + namelist[index] + '_SW.rpb')
            if os.path.exists(filepath):
                shutil.copy(filepath, target_path)
            if os.path.exists(rpbpath):
                shutil.copy(rpbpath, target_path)
            print(filepath + ' is finished')


def move_rpbs():
    target_path = "I:\\AHSI_result"
    filefolder_list = ["F:\\AHSI_part1", "H:\\AHSI_part2", "L:\\AHSI_part3", "I:\\AHSI_part4"]
    for filefolder in filefolder_list:
        filelist, namelist = get_subdirectories(filefolder)
        # 遍历每一个数据文件夹 并进行处理
        for index in range(len(filelist)):
            # 获取 SW波段的文件路径
            rpbpath = os.path.join(filelist[index], namelist[index] + '_SW.rpb')
            if os.path.exists(rpbpath):
                shutil.copy(rpbpath, target_path)
            print(rpbpath + ' is finished')


def image_coordinate(image_path: str):
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
            corrected_image_path = image_path.replace('SW.tif', "_Corrected.tif")
            warp_options = gdal.WarpOptions(rpc=True)
            corrected_dataset = gdal.Warp(str(corrected_image_path), dataset, options=warp_options)
            corrected_dataset = None
            dataset = None
            print("Correction completed.")
    except FileNotFoundError as fnf_error:
        print(fnf_error)


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


testfile = "I:\AHSI_result\GF5B_AHSI_W104.3_N32.3_20220209_002267_L10000074985_SW.tif"
image_coordinate(testfile)

# move_previous_result()
# folder_path = r"F:\ahsi"
# sub, filename = get_subdirectories(folder_path)
# total = zip(sub, filename)
# for folder, name in total:
#     image_path = os.path.join(folder, name + '_VN.tif')
#     dataset = gdal.Open(image_path)
#     image = dataset.ReadAsArray()
#     rgb = image[[59, 38, 20], :, :]
#     array_to_dataset(rgb, image_path)
