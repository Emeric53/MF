import numpy as np

filepath = "H:\\重新下载数据\\GF5B_AHSI_W102.8_N32.3_20220424_003345_L10000118222\\GF5B_AHSI_Spectralresponse_SWIR.raw"
data = np.genfromtxt(filepath)


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
    f.write("Nanometer data for EMIT (assume Gaussian with maximum response of 1)\n")

# for k in  range(len(fwhm)):
for k in range(len(fwhm)):
    center = wavelength[k]
    thisfwhm = fwhm[k]
    waveband = np.linspace(center - 9, center + 9, 51)
    response = gaussian_response(waveband, center, thisfwhm)
    with open('response.txt', 'a') as f:
        f.write("CENTER: {:.4f} NM    FWHM: {:.3f} NM\n".format(center, thisfwhm))
        for i in range(len(response)):
            f.write('{:.4f} {:.7f} {:.2f}\n'.format(waveband[i], response[i], 10000000 / waveband[i]))
    f.close()
