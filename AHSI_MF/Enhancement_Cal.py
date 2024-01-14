import numpy as np
from osgeo import gdal
import sys
# 本代码：利用匹配滤波算法计算甲烷增强的浓度，单位为 ppm*m 并转换为 单位为ppm的混合比增强 输出为tiff影像

# 设置初始栅格文件路径
file_path = "C:\\Users\\RS\\Desktop\\output\\test.tif"

# 利用gdal打开数据
dataset = gdal.Open(file_path, gdal.GA_ReadOnly)

# 从已存在的TIFF文件中获取地理参考信息
geo_transform = dataset.GetGeoTransform()
projection = dataset.GetProjection()

# 获取所有波段数目
num_bands = dataset.RasterCount

# 定义数组存储各波段数据,以及波段中非nan的相关个数
band_data_list = []

# 遍历各个波段，注意数据集的getrasterband方法索引从1开始
for band_index in range(1, num_bands + 1):
    # 依据索引获取当前波段数据
    current_band = dataset.GetRasterBand(band_index)
    # 读取波段数据为NumPy数组
    current_band_data = current_band.ReadAsArray()
    # 添加入波段数据总和中
    band_data_list.append(current_band_data)

# 将波段数据转为np数组array
image_data = np.array(band_data_list)

# 获取栅格数据的波段，行，列
bands, rows, cols = image_data.shape

# 打开AHSI的单位吸收光谱文件并转换为numpy数组
unitabsorptionspectrum = []

with open('AHSI_MF\\unit_absorption_spectrum.txt', 'r') as file:
    data = file.readlines()
    for i in data:
        split_i = i.split(' ')
        absorption = split_i[1].rstrip('\n')
        unitabsorptionspectrum.append(float(absorption))

unitabsorptionspectrum = np.array(unitabsorptionspectrum)

# 构造总的甲烷浓度增强  以及用于地表反照率校正的二维数组变量
total_alpha = np.zeros((rows, cols))
albedo = np.zeros((rows, cols))

# 根据原始数据光谱计算背景光谱以及协方差矩阵
for col in range(cols):
    # 构造初始协方差矩阵变量
    c = np.zeros((bands, bands))
    # 进行数据的筛选获得每一列的数据
    trackdata = image_data[:, :, col]
    # 沿着波段维度计算均值
    mean_along_band = np.mean(trackdata, axis=1)
    # 得到这一行的均值背景光谱
    for row in range(rows):
        if not np.isnan(image_data[0, row, col]):
            c += np.outer(image_data[:, row, col] - mean_along_band,
                          image_data[:, row, col] - mean_along_band)
    c = c / rows
    # 获取协方差矩阵的逆矩阵
    c_inverse = np.linalg.inv(c)
    # 基于背景光谱和单位吸收光谱获得初始目标谱
    target_spectrum = np.multiply(mean_along_band, unitabsorptionspectrum)
    # 基于初始目标谱和匹配滤波算法公式 计算对应列的初始的甲烷浓度增强
    for row in range(rows):
        #计算 per-pixel的 地表反照率校正项
        albedo[row, col] = (np.inner(image_data[:, row, col], mean_along_band)
                            / np.inner(mean_along_band, mean_along_band))
        #依靠优化问题的解 计算甲烷浓度的增强
        if not np.isnan(image_data[0, row, col]):
            up = ((image_data[:, row, col] - mean_along_band) @ c_inverse
                  @ target_spectrum)
            down = albedo[row, col] * target_spectrum @ c_inverse @ target_spectrum
            total_alpha[row, col] = up / down
        else:
            total_alpha[row, col] = np.nan
#构造用于稀疏校正的l1校正项 数组
l1filter = np.zeros((rows, cols))
#进行迭代运算
for i in range(20):
    # 获得去除了前一步计算的甲烷增强影像的卫星光谱
    iter_data = image_data.copy()
    for row in range(rows):
        for col in range(cols):
            l1filter[row,col] = 1/(total_alpha[row, col] + sys.float_info.epsilon)
            if not np.isnan(image_data[0, row, col]):
                iter_data[:, row, col] = (image_data[:, row, col] -
                                          albedo[row, col] * total_alpha[row, col] *
                                          target_spectrum)
    # 更新背景光谱以及协方差矩阵
    for col in range(cols):
        # 构造初始协方差矩阵变量
        c = np.zeros((bands, bands))
        # 进行数据的筛选获得每一列的数据
        trackdata = iter_data[:, :, col]
        # 每一行 进行一次遍历
        # 沿着波段维度计算均值
        mean_along_band = np.mean(trackdata, axis=1)
        target_spectrum = np.multiply(mean_along_band, unitabsorptionspectrum)
        for row in range(rows):
            if not np.isnan(image_data[0, row, col]):
                c += np.outer(image_data[:, row, col] - (mean_along_band +
                                                         albedo[row, col] * total_alpha[
                                                             row, col] * target_spectrum),
                              image_data[:, row, col] - (mean_along_band +
                                                         albedo[row, col] * total_alpha[
                                                             row, col] * target_spectrum))
        c = c / rows
        c_inverse = np.linalg.inv(c)
        # 计算新的甲烷浓度增强
        for row in range(rows):
            if not np.isnan(image_data[0, row, col]):
                up =((image_data[:, row, col] - mean_along_band) @ c_inverse @ target_spectrum )- l1filter[row, col]
                down = albedo[row, col]*target_spectrum @ c_inverse @ target_spectrum
                total_alpha[row, col] = max(up / down, 0)
            else:
                total_alpha[row, col] = np.nan

total_alpha = total_alpha * 0.00125
# 指定输出的TIFF文件名
output_tiff_file = "C:\\Users\\RS\\Desktop\\output\\ahsi_methane\\albedo_adjust_percolumn_iter_20_output.tiff"
# 获取数组的维度
rows, cols = total_alpha.shape

# 创建一个新的TIFF文件
driver = gdal.GetDriverByName('GTiff')
dataset = driver.Create(output_tiff_file, cols, rows, 1, gdal.GDT_Float32)

# 将NumPy数组写入TIFF文件
band = dataset.GetRasterBand(1)
band.WriteArray(total_alpha)

# 设置获取的地理参考信息
dataset.SetGeoTransform(geo_transform)
dataset.SetProjection(projection)
