"""this code is used to process the radiance file by using the matching filter algorithm
and the goal is to get the methane enhancement image"""

# the necessary lib to be imported
import numpy as np
from osgeo import gdal
import pathlib as pl

# a function to get the raster array and return the dataset
def get_raster_array(filepath):
    # 利用gdal打开数据
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    return dataset

# a function to open the unit absorption spectrum file and returen the numpy array
def open_unit_absorption_spectrum(filepath):
    # 打开AHSI的单位吸收光谱文件并转换为numpy数组
    unitabsorptionspectrum = []
    with open(filepath, 'r') as file:
        data = file.readlines()
        for band in data:
            split_i = band.split(' ')
            band = split_i[1].rstrip('\n')
            unitabsorptionspectrum.append(float(band))
    output = np.array(unitabsorptionspectrum)
    return output

# difine the path of the unit absorption spectrum file and open it
uas_filepath = 'EMIT_unit_absorption_spectrum.txt'
unitabsorptionspectrum = open_unit_absorption_spectrum(uas_filepath)

# define the path of the radiance folder and get the radiance file list with an img suffix
radiance_folder = "F:\\EMIT_DATA\\envi"
radiance_path_list = pl.Path(radiance_folder).glob('*.img')

# get the output file path and get the existing output file list to avoid the repeat process
root = pl.Path("F:\\EMIT_DATA\\result")
output = root.glob('*.tif')
outputfile = []
for i in output:
    outputfile.append(str(i.name))

# define the main function to process the radiance file by using the matching filter algorithm
# the input includes the radiance file path, the unit absorption spectrum, the output path and the is_iterate flag
def mf_process(filepath, unitabsorptionspectrum, output_path, is_iterate=False):
    # get the file name to make the output path string
    name = filepath.name.rstrip("radiance.img")

    # get the raster array from the radiance file
    dataset = get_raster_array(str(filepath))

    # get the basic information of the raster file such as the geo transform, the projection and the number of bands
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    num_bands = dataset.RasterCount

    # pre-define the list to store the band data and the count of the non-nan value
    band_data_list = []
    count_not_nan = 0

    # iterate the bands to get the band data and the count of the non-nan value
    for band_index in range(0, len(unitabsorptionspectrum)):
        # 依据索引获取波段数据
        band = dataset.GetRasterBand(num_bands - len(unitabsorptionspectrum) + band_index + 1)
        # 读取波段数据为NumPy数组
        band_data = band.ReadAsArray()
        # 添加入波段数据总和中
        band_data_list.append(band_data)
        # 计算非nan像元的个数
        count_not_nan = np.count_nonzero(~np.isnan(band_data))

    # 将波段数据转为np array
    image_data = np.array(band_data_list)

    # 以每个波段为分割，计算非nan的平均值 作为背景光谱
    u = np.nanmean(image_data, axis=(1, 2))

    # 获取遥感影像的波段数，行，列
    bands, rows, cols = image_data.shape

    # 构造总的甲烷浓度增强 地表反照率校正 用于稀疏校正的l1校正项 的二维数组变量
    albedo = np.zeros((rows, cols))
    alpha = np.zeros((rows, cols))
    l1filter = np.zeros((rows, cols))

    # 构造协方差矩阵
    c = np.zeros((bands, bands))
    for row in range(rows):
        for col in range(cols):
            if not np.isnan(image_data[0, row, col]):
                c += np.outer(image_data[:, row, col] - u, image_data[:, row, col] - u)
    c = c / count_not_nan
    # 取协方差矩阵的逆矩阵
    c_inverse = np.linalg.inv(c)

    # 基于单位吸收光谱和背景值 计算目标谱
    target = np.multiply(u, unitabsorptionspectrum)

    # 空间上遍历整个遥感影像
    for row in range(rows):
        for col in range(cols):
            # 计算甲烷浓度增强初步值
            if not np.isnan(image_data[0, row, col]):
                # 计算地表反照率改正项
                albedo[row, col] = (np.inner(image_data[:, row, col], u)
                                    / np.inner(u, u))
                # 计算甲烷浓度增强值
                up = (image_data[:, row, col] - u) @ c_inverse @ target
                down = albedo[row, col] * (target @ c_inverse @ target)
                alpha[row, col] = up / down
            else:
                alpha[row, col] = np.nan
    if is_iterate:
        # 迭代计算甲烷浓度增强
        for i in range(20):
            iter_data = image_data.copy()
            for row in range(rows):
                for col in range(cols):
                    if not np.isnan(image_data[0, row, col]):
                        iter_data[:, row, col] = image_data[:, row, col] - target * alpha[row, col]
            # 更新背景光谱和目标谱
            u = np.nanmean(iter_data, axis=(1, 2))
            target = np.multiply(u, unitabsorptionspectrum)
            # 更新协方差矩阵
            c = np.zeros((bands, bands))
            for row in range(rows):
                for col in range(cols):
                    if not np.isnan(image_data[0, row, col]):
                        c += np.outer(image_data[:, row, col] - (u + albedo[row, col] * alpha[row, col] * target),
                                      image_data[:, row, col] - (u + albedo[row, col] * alpha[row, col] * target))
            c = c / count_not_nan

            # 取协方差矩阵的逆矩阵
            c_inverse = np.linalg.inv(c)

            # 空间上遍历整个遥感影像
            for row in range(rows):
                for col in range(cols):
                    if not np.isnan(image_data[0, row, col]):
                        # 计算新的甲烷浓度增强值
                        up = (image_data[:, row, col] - u) @ c_inverse @ target
                        down = albedo[row, col] * target @ c_inverse @ target
                        alpha[row, col] = max(up / down, 0)
                    else:
                        alpha[row, col] = np.nan

    # 指定输出路径
    output_tiff_file = str(output_path / (name + 'enhancement.tif'))

    # 获取数组的维度
    rows, cols = alpha.shape

    # 创建一个新的TIFF文件
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_tiff_file, cols, rows, 1, gdal.GDT_Float32)

    # 将NumPy数组写入TIFF文件
    band = dataset.GetRasterBand(1)
    band.WriteArray(alpha)

    # 设置获取的地理参考信息
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    dataset = None


for radiance_path in radiance_path_list:
    currentfilename = str(radiance_path.name.rstrip("radiance.img") + "enhancement.tif")
    if currentfilename in outputfile:
        continue
    else:
        print(f"{currentfilename} is now being processed")
        try:
            mf_process(radiance_path, unitabsorptionspectrum, root, is_iterate=False)
            print(f"{currentfilename} has been processed")
        except Exception:
            print(f"{currentfilename} has failed to process")
            pass

