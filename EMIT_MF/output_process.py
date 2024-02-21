"""该代码用于对计算得到的柱浓度增强数据进行中值滤波和去噪等处理"""
import numpy as np
import cv2

# 设置读取文件路径
filepath = "C:\\Users\\RS\\Desktop\\EMIT\\Result\\EMIT_L1B_RAD_001_20230204T041009_2303503_016_Enhancement.tiff"

# 读取TIFF格式的图像
image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

# 对灰度图进行高斯滤波去噪处理
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# 计算整幅图像的中值
threshold_value = np.median(blurred_image)

# 创建一个新的图像，低于中值的像素值设为nodata，高于中值的像素值减去中值
processed_image = np.where(blurred_image < threshold_value, 0,blurred_image)

# 保存处理后的图像
cv2.imwrite("C:\\Users\\RS\\Desktop\\EMIT\\Result\\EMIT_20230204T041009_Enhancement_processed.tiff", processed_image)
