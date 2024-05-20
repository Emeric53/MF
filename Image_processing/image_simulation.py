import numpy as np
from osgeo import gdal

# load the simulated modtran spectrum
simulated_spectrum = np.load("C:\\Users\\RS\\Desktop\\EMIT_band_radiance.npy")

# load the real spectrum
with open(r"C:\Users\RS\Desktop\All\EMIT_wavelengths.csv", 'r') as wvl:
    central_wvl = wvl.readline().rstrip('\n').split(',')
    central_wvl = np.array(central_wvl)
    center_wavelengths = central_wvl.astype(np.float32)

# filter the spectrum and center_wavelengths for non value
center_wavelengths = center_wavelengths[simulated_spectrum > 0]
simulated_spectrum = simulated_spectrum[simulated_spectrum > 0]

# set the shape of the image that want to simulate
band_num = len(simulated_spectrum)
row_num = 200
col_num = 200

# generate the universal radiance image
simulated_image = simulated_spectrum.reshape(band_num, 1, 1) * np.ones([row_num, col_num])

# generate the noisy image with the same size of the universal radiance image
simulated_noisyimage = np.zeros_like(simulated_image)

# set the SNR for the simulated image
SNR = 300

# add the gaussian noise to the image
for i in range(simulated_image.shape[0]):  # 遍历每个波段
    mu = np.mean(simulated_image[i])  # 计算当前波段的平均亮度
    sigma = mu / np.sqrt(SNR)  # 计算对应的噪声标准差
    print(sigma)
    noise = np.random.normal(0, sigma, (row_num, col_num))  # 生成高斯噪声
    print(noise)
    simulated_noisyimage[i] = simulated_image[i] + noise  # 添加噪声到原始数据
print(simulated_noisyimage)
print(simulated_noisyimage.shape)
# set the mask for the image
transmission_mask = np.ones(simulated_noisyimage.shape)
plume = np.ones([50, 50]) * 0.1
origin = [5, 6]  # upper left corner of the plume
transmission_mask[:, origin[0] - 1:origin[0] + plume.shape[0] - 1,
origin[1] - 1:origin[1] + plume.shape[1] - 1] *= plume
final_simulation = transmission_mask * simulated_noisyimage

# 创建一个新的GDAL内存数据集
driver = gdal.GetDriverByName('GTiff')
dataset = driver.Create("C:\\Users\\RS\\Desktop\\simulation.tiff", final_simulation.shape[2], final_simulation.shape[1],
                        final_simulation.shape[0], gdal.GDT_Float32)

# # 设置地理变换（如果需要）
# # 这里只是一个示例，按需修改
# dataset.SetGeoTransform([0, 1, 0, 0, 0, -1])  # depends on the real case
#
# # 设置空间参考（如果需要）
# # 创建一个空间参考对象
# srs = osr.SpatialReference()
# srs.ImportFromEPSG(4326)  # WGS84 lat/lon
# dataset.SetProjection(srs.ExportToWkt())

# 写入数据
for index in range(final_simulation.shape[0]):
    dataset.GetRasterBand(index + 1).WriteArray(final_simulation[index])

# 保存并关闭数据集
dataset.FlushCache()
dataset = None
