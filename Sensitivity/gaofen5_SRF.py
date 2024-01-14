'''该代码用于生成光谱响应函数'''
import numpy as np

#读取文件 获取传感器的中心波长和 FWHM
filepath = "C:\\Users\\RS\\Desktop\\GF5B_AHSI_Spectralresponse_SWIR.txt"

# 初始化两个空数组用于存储数据
column1 = []
column2 = []



# 打开文本文件
with open(filepath, 'r') as file:
    lines = file.readlines()

# 分离每一行的两列数据并存储到相应的数组中
for line in lines:
    data = line.strip().split(',')
    if len(data) == 2:
        column1.append(float(data[0]))
        column2.append(float(data[1]))

# 转换数组为NumPy数组
wavelength = np.array(column1)
fwhm = np.array(column2)

def gaussian_response(wavelengths, center_wavelength, fwhm):
    """
    Calculate the Gaussian response for a given channel.
    Parameters:
        wavelengths (array): Array of wavelengths to calculate the response for.
        center_wavelength (float): The center wavelength of the channel.
        fwhm (float): Full Width at Half Maximum for the channel.

    Returns:
        response (array): The Gaussian response of the channel at the given wavelengths.
    """
    # Calculate standard deviation from FWHM
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    # Calculate the Gaussian response
    response = np.exp(-((wavelengths - center_wavelength) ** 2) / (2 * sigma ** 2))
    return response


with open('response.txt', 'w') as f:
    f.write("Nanometer data for GaoFen5B (assume Gaussian with maximum response of 1)\n")

for k in range(len(fwhm)):
    center = wavelength[k]
    thisfwhm = fwhm[k]
    waveband = np.linspace(center-9, center+9,51)
    response = gaussian_response(waveband, center, thisfwhm)
    with open('response.txt', 'a') as f:
        f.write("CENTER: {:.4f} NM    FWHM: {:.3f} NM\n".format(center, thisfwhm))
        for i in range(len(response)):
            f.write('{:.4f} {:.7f} {:.2f}\n'.format(waveband[i], response[i], 10000000/waveband[i]))
    f.close()
