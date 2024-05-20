"""
    该代码 读取 GF5B的光谱响应文件，然后计算高斯响应函数，最后将结果写入文件。

    Returns:
        _type_: _description_
"""
import numpy as np

# 读取 GF5B的光谱响应文件
filepath = "AHSI_MF\GF5B_AHSI_Spectralresponse_SWIR.raw"
response = []
with open('AHSI_MF\GF5B_AHSI_RadCal_SWIR.raw', 'r') as radiation_calibration_file:
    result = radiation_calibration_file.readlines()
    for i in result:
        wvl = i.split(',')[0]
        res = i.split(',')[1].rstrip('\n')
        response.append([wvl, res])
print(response)


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
    thisfwhm = fwhm[k]
    waveband = np.linspace(center - 9, center + 9, 51)
    response = gaussian_response(waveband, center, thisfwhm)
    with open('response.txt', 'a') as f:
        f.write("CENTER: {:.4f} NM    FWHM: {:.3f} NM\n".format(center, thisfwhm))
        for i in range(len(response)):
            f.write('{:.4f} {:.7f} {:.2f}\n'.format(waveband[i], response[i], 10000000 / waveband[i]))
    f.close()
