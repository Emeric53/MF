import numpy as np
from osgeo import gdal
import sys
import cv2

# 读取TIFF格式的图像
image = cv2.imread("C:\\Users\\RS\\Desktop\\output\\ahsi_methane\\albedo_adjust_percolumn_iter_20_output.tiff",
                   cv2.IMREAD_UNCHANGED)

# 对灰度图进行高斯滤波去噪处理
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# 计算整幅图像的中值
threshold_value = np.median(blurred_image)

# 创建一个新的图像，低于中值的像素值设为nodata，高于中值的像素值减去中值
processed_image = np.where(blurred_image < threshold_value, 0,
                           blurred_image - threshold_value)
# 保存处理后的图像
cv2.imwrite("C:\\Users\\RS\\Desktop\\output\\ahsi_methane\\processed_output.tiff", processed_image)
