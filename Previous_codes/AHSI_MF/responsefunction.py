import numpy as np

# 读取 GF5B的光谱响应文件
with open('AHSI_MF\\GF5B_AHSI_Spectralresponse_SWIR.raw', 'r') as radiation_calibration_file:
    result = radiation_calibration_file.readlines()
    response = [[line.strip().split(',')[0], line.strip().split(',')[1]] for line in result]


def gaussian(wavelengths, center, fwhm):
    """生成高斯分布函数。

    参数：
    wavelengths -- 波长数组
    center -- 中心波长
    fwhm -- 全宽半最大值

    返回：
    高斯分布的响应值
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)


def spectral_response_function(centers, fwhms, wavelength_range):
    """生成卫星传感器的光谱响应函数（高斯模型）。

    参数：
    centers -- 中心波长数组
    fwhms -- 对应的FWHM数组
    wavelength_range -- 波长范围数组

    返回：
    波长范围内的光谱响应函数数组
    """
    response = np.zeros_like(wavelength_range)
    for center, fwhm in zip(centers, fwhms):
        response += gaussian(wavelength_range, center, fwhm)
    return response


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
    f.write("Nanometer data for AHSI (assume Gaussian with maximum response of 1)\n")

# for k in  range(len(fwhm)):
for k in range(len(response)):
    center = response[k][0]
    thisfwhm = response[k][1]
    waveband = np.linspace(center - 9, center + 9, 51)
    response = gaussian_response(waveband, center, thisfwhm)
    with open('response.txt', 'a') as f:
        f.write("CENTER: {:.4f} NM    FWHM: {:.3f} NM\n".format(center, thisfwhm))
        for i in range(len(response)):
            f.write('{:.4f} {:.7f} {:.2f}\n'.format(waveband[i], response[i], 10000000 / waveband[i]))
    f.close()
