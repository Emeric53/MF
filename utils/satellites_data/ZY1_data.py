from osgeo import gdal
import numpy as np


# !对 ZY1 上的AHSI dat数据读取SZA和地面高程
def get_sza_altitude(filepath: str):
    hdr_file = filepath.replace(".dat", ".hdr")
    with open(hdr_file, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=")
                if key.strip() == "solar zenith":
                    sza = float(value.strip())
                    print(f"SZA: {sza}")
    return sza, 0


# 获取ahsi的波段信息
def get_ZY1_ahsi_bands():
    """
    get bands list of ahsi
    :param band_file:  filepath containing bands wavelength
    :return: bands list
    """
    # 读取校准文件
    wavelengths = np.load(
        "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\data\\satellite_channels\\ZY1_channels.npz"
    )["central_wvls"]
    return wavelengths


def get_ZY1_radiances_from_dat(dat_file, low, high):
    radiance_cube = gdal.Open(dat_file)
    radiance = radiance_cube.ReadAsArray()
    wvls = get_ZY1_ahsi_bands()
    indices = np.where((wvls >= low) & (wvls <= high))[0]
    radiance = radiance[indices, :, :]
    return wvls[indices], radiance


def extract_wavelengths_from_hdr(hdr_file):
    wavelengths = []
    inside_wavelength_section = False

    # 打开并逐行读取 .hdr 文件
    with open(hdr_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("wavelength"):
                inside_wavelength_section = True
                continue
            if inside_wavelength_section:
                if line.startswith("{"):
                    # 波长数据的开始
                    continue
                elif line.startswith("}"):
                    # 波长数据的结束
                    break
                else:
                    # 添加波长值
                    wavelengths.extend([float(x) for x in line.split(",")])

    return np.array(wavelengths)


if __name__ == "__main__":
    zy1_hdr = "J:\stanford\ZY1\ZY1F_AHSI_W111.72_N33.06_20221026_004370_L1A0000265656_VNSW_Rad.hdr"
    zy1_dat = "J:\stanford\ZY1\ZY1F_AHSI_W111.72_N33.06_20221026_004370_L1A0000265656_VNSW_Rad.dat"
    radiance = get_ZY1_radiances_from_dat(zy1_dat, 2150, 2500)
    print(radiance.shape)
