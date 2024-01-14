#基于该代码实现 对AHSI的辐射定标工作。
from osgeo import gdal

# 设置初始栅格文件路径
file_path = "H:\\高分5号02星\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222_SW.tif"
#利用gdal打开数据
dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
# 从已存在的TIFF文件中获取地理参考信息
geo_transform = dataset.GetGeoTransform()
projection = dataset.GetProjection()

factorlist=[]
with open("C:\\Users\\RS\\Desktop\\GF5B_AHSI_RadCal_SWIR.txt", 'r') as radiation_calibration_file:
    result = radiation_calibration_file.readlines()
    for i in result:
        factor = i.split(',')[0]
        factorlist.append(factor)
print(factorlist)

# 获取所有波段数目
num_bands = dataset.RasterCount
print(num_bands)
# 定义数组存储各波段数据,以及波段中非nan的相关个数
band_data_list = []
count_non_nan = 0
# 遍历各个波段，注意数据集的getrasterband方法索引从1开始
for band_index in range(1, 20):
    # 依据索引获取当前波段数据
    current_band = dataset.GetRasterBand(band_index)
    # 读取波段数据为NumPy数组
    current_band_data = current_band.ReadAsArray()*float(factorlist[band_index-1])
    # 添加入波段数据总和中
    band_data_list.append(current_band_data)
