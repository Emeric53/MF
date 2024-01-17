"""该代码用于从甲烷增强影像中提取出特定烟羽的整体形状"""
from osgeo import gdal
import numpy as np
import scipy.stats as stats



# get the path of the methane enhancement images
def get_raster_array(filepath):
    # 利用gdal打开数据
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    return dataset

observation = get_raster_array("C:\\Users\\RS\\Desktop\\EMIT\\Result\\EMIT_20230204T041009_Enhancement_clip.tiff.tif")

# 从已存在的TIFF文件中获取地理参考信息以及波段信息
geo_transform = observation.GetGeoTransform()
projection = observation.GetProjection()
bg = get_raster_array("C:\\Users\\RS\\Desktop\\EMIT\\Result\\EMIT_20230204T041009_Enhancement_clip_bg.tiff.tif")
originate_point = []
# 定义padding的大小
padding_size = 2  # 5*5窗口的一半大小为2
# 读取遥感影像数据
image_data = observation.ReadAsArray()
bg_data = bg.ReadAsArray()


# 对遥感影像进行padding
padded_image = np.pad(image_data, ((padding_size, padding_size), (padding_size, padding_size)), mode="symmetric")
rows, cols = padded_image.shape

mask = np.zeros((rows-4, cols-4))
for row in range(rows-4):
    for col in range(cols-4):
        pixel_window = padded_image[row:row+5, col:col+5]
        # 假设您有两组样本数据 a 和 b
        t_statistic, p_value = stats.ttest_ind(bg_data.flatten(), pixel_window.flatten(), equal_var=False)
        if p_value <= 0.005:
            mask[row, col] = 1

#中值空间滤波
padded_mask = np.pad(mask, ((1, 1), (1, 1)), mode="symmetric")
for row in range(padded_mask.shape[0]-2):
    for col in range(padded_mask.shape[1]-2):
        pixel_window = padded_mask[row: row+3, col: col+3]
        mask[row, col] = np.median(pixel_window.flatten())
#高斯卷积

# 指定输出的TIFF文件名
output_tiff_file = "C:\\Users\\RS\\\Desktop\\EMIT\\Result\\mask.tiff"
# 获取数组的维度
rows, cols = mask.shape

# 创建一个新的TIFF文件
driver = gdal.GetDriverByName('GTiff')
dataset = driver.Create(output_tiff_file, cols, rows, 1, gdal.GDT_Float32)

# 将NumPy数组写入TIFF文件
band = dataset.GetRasterBand(1)
band.WriteArray(mask)

# 设置获取的地理参考信息
dataset.SetGeoTransform(geo_transform)
dataset.SetProjection(projection)
