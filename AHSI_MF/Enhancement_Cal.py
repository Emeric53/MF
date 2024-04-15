#based on the matching filter algorithm to process the AHSI data and get the methane enhancement result
import pathlib as pl
import numpy as np
from osgeo import gdal
import shutil
import os


# a function to get the raster array and return the dataset
def get_raster_array(filepath):
    # open the dataset from the filepath
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    # 获取波段数
    band_count = dataset.RasterCount

    # 创建一个 NumPy 数组来存储所有波段的数据
    data_array = np.zeros((band_count, dataset.RasterYSize, dataset.RasterXSize))

    # 读取所有波段的数据
    for i in range(1, band_count + 1):
        band = dataset.GetRasterBand(i)
        data_array[i - 1] = band.ReadAsArray()
    return data_array


# operate the radiation calibration on the AHSI l1 data
def rad_calibration(dataset):
    with open("GF5B_AHSI_RadCal_SWIR.raw", "r") as file:
        lines = file.readlines()
        index = 0
        for line in lines:
            a = float(line.split(',')[0])
            b = float(line.split(",")[1].rstrip('\n'))
            dataset[index, :, :] = a * dataset[index, :, :] + b
            index += 1
    return dataset


# open the unit absorption spectrum
def open_unit_absorption_spectrum(filepath, min, max):
    # open the unit absorption spectrum file and convert it a numpy array
    uas_list = []
    with open(filepath, 'r') as file:
        data = file.readlines()
        for band in data:
            split_i = band.split(' ')
            wvl = float(split_i[0])
            if min <= wvl <= max:
                band = split_i[1].rstrip('\n')
                uas_list.append(float(band))
    out_put = np.array(uas_list)
    return out_put


# export the result array to tiff file
def export_array2tiff(result, filepath, outputfolder):
    filename = pl.Path(filepath).name
    outputpath = outputfolder + "/" + filename

    # open the dataset from the filepath
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    # 从已存在的TIFF文件中获取地理参考信息
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    # 创建一个 GDAL 数据集
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(outputpath, result.shape[1], result.shape[0], 1, gdal.GDT_Float32)

    # 设置空间参考信息
    dataset.SetProjection(projection)
    dataset.SetGeoTransform(geo_transform)

    # 将 NumPy 数组写入 GDAL 数据集
    dataset.GetRasterBand(1).WriteArray(result)

    # 关闭 GDAL 数据集
    dataset = None

    image_coordinate(outputpath)


# use rpb file to get the projection for the tiff file
def image_coordinate(image_path):
    dataset = gdal.Open(image_path)
    if dataset.GetMetadata('RPC') is None:
        print("RPC file is not found.")
    else:
        corrected_image_path = image_path.replace('.tif', '_corrected.tif')
        warp_options = gdal.WarpOptions(rpc=True)
        corrected_dataset = gdal.Warp(corrected_image_path, dataset, options=warp_options)
        corrected_dataset = None
        dataset = None
        print("校正完成")


# define the main function to process the radiance file by using the matching filter algorithm
def mf_process(filepath, uas_path, output_path, is_iterate=False, is_albedo=False, is_filter=False):
    # open the unit absorption spectrum file and convert it a numpy array
    unit_absorption_spectrum = open_unit_absorption_spectrum(uas_path, 2100, 2500)
    # get the raster array from the radiance file
    radiance_data = rad_calibration(get_raster_array(filepath))
    radiance_window = radiance_data[radiance_data.shape[0] - len(unit_absorption_spectrum) - 1:-1, :, :]
    # transpose the array to band row col organization.
    radiance_window = np.transpose(radiance_window, [1, 2, 0])
    name = pl.Path(filepath).name.rstrip(".tif")
    # pre-define the list to store the band data and the count of the non-nan value
    # get the number of bands, rows and columns of the image data
    rows, cols, bands = radiance_window.shape
    albedo = np.zeros((rows, cols))
    alpha = np.zeros((rows, cols))
    # calculate the covariance matrix
    c = np.zeros((bands, bands))

    # alone the row axis to get the mean value of each band for each column
    for col_index in range(cols):
        # get the data of each column
        column_data = radiance_window[:, col_index, :]
        # count the nan number in the column data
        count_not_nan = np.count_nonzero(~np.isnan(column_data))
        if count_not_nan == 0:
            alpha[:, col_index] = np.nan
            continue
        # get the mean value of each band for each column
        u = np.nanmean(column_data, axis=0)
        # based on the background spectrum and the unit absorption spectrum, calculate the target spectrum
        target = np.multiply(u, unit_absorption_spectrum)

        # alone the row axis to calculate the covariance matrix
        for row in range(rows):
            if not np.isnan(column_data[row, 0]):
                c += np.outer(column_data[row, :] - u, column_data[row, :] - u)

        # get the average of the sum of covariance matrix
        c = c / count_not_nan

        # get the inverse of the covariance matrix
        c_inverse = np.linalg.inv(c)

        # iterate the whole region of the image data to calculate the albedo and the alpha for each pixel
        for row_index in range(rows):
            # account the nan value in the image data and make the result to be nan
            if not np.isnan(column_data[row_index, 0]):
                if is_albedo:
                    # based on the formula to calculate the albedo of each pixel and store it into the albedo array
                    albedo[row_index, col_index] = np.inner(column_data[row_index, :], u) / np.inner(u, u)
                else:
                    albedo[row_index, col_index] = 1
                # based on the formula to calculate the methane enhancement of each pixel
                # and store it in the alpha array
                up = (column_data[row_index, :] - u) @ c_inverse @ target
                down = albedo[row_index, col_index] * (target @ c_inverse @ target)
                alpha[row_index, col_index] = up / down
            else:
                alpha[row_index, col_index] = np.nan
        if is_iterate:
            # build the l1_filter to store the result
            l1filter = np.zeros((rows, cols))
            # define a tiny value to avoid the zero division
            epsilon = np.finfo(np.float32).tiny
            # based on the former alpha result, update the background spectrum and the target spectrum
            # and then update the covariance matrix
            # and get the new methane enhancement result
            iter_data = column_data.copy()
            for iter_num in range(4):
                for row_index in range(rows):
                    if not np.isnan(column_data[row_index, 0]):
                        iter_data[row_index, :] = column_data[row_index, :] - albedo[row_index, col_index] * target * \
                                                  alpha[row_index, col_index]
                        if is_filter:
                            l1filter[row_index, col_index] = 1 / (alpha[row_index, col_index] + epsilon)
                        else:
                            l1filter[row_index, col_index] = 0

                count_not_nan = np.count_nonzero(~np.isnan(iter_data[:, 0]))
                u = np.nanmean(iter_data, axis=0)
                target = np.multiply(u, unit_absorption_spectrum)
                # get the new covariance matrix
                c = c * 0
                for row_index in range(rows):
                    if not np.isnan(iter_data[row_index, 0]):
                        c += np.outer(iter_data[row_index, :] - u, iter_data[row_index, :] - u)
                c = c / count_not_nan
                # get the inverse of the covariance matrix
                c_inverse = np.linalg.inv(c)

                # calculate the new methane enhancement result
                for row_index in range(rows):
                    if not np.isnan(column_data[row_index, 0]):
                        up = (column_data[row_index, :] - u) @ c_inverse @ target - l1filter[row_index, col_index]
                        down = albedo[row_index, col_index] * target @ c_inverse @ target
                        alpha[row_index, col_index] = max(up / down, 0)
                    else:
                        alpha[row_index, col_index] = np.nan

    # set the output tiff file path
    #  get the number of rows and columns of the alpha array
    output_folder = str(output_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    rpb_file = str(filepath.replace('.tif', '.rpb'))
    shutil.copy(rpb_file,output_folder)
    # use the function to export the methane enhancement result to a nc file
    export_array2tiff(alpha, filepath, output_folder)


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


if '__main__' == __name__:

    filefolder = "F:\\ahsi"
    filelist,namelist = get_subdirectories(filefolder)
    for index in range(len(filelist)):
        filepath = os.path.join(filelist[index], namelist[index]+'_SW.tif')
        outputfolder = os.path.join(filelist[index], 'result')
        outputfile = os.path.join(outputfolder, namelist[index]+'_SW.tif')
        if os.path.exists(outputfile):
            pass
        else:
            print(namelist[index] + ' is processing')
            try:
                mf_process(filepath,"unit_absorption_spectrum.txt", outputfolder, False)
            except Exception as e:
                print("ERROR")

#
# # define the path of the unit absorption spectrum file and open it
# uas_filepath = 'unit_absorption_spectrum.txt'
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
