import numpy as np
import xarray as xr
import h5py


filename = (
    "I:\\PRISMA_控制释放实验\\PRS_L1_STD_OFFL_20221015181614_20221015181618_0001.he5"
)


# 打开 HE5 文件
def get_prisma_hierachy(filename):
    with h5py.File(filename, "r") as f:
        # 定义一个函数来打印每个数据集的维度信息
        def print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"数据集名称: {name}")
                print(f"数据维度: {obj.shape}")
                print(f"数据类型: {obj.dtype}")
                print("-" * 30)

        # 递归访问所有数据集并打印信息
        f.visititems(print_dataset_info)


def get_prisma_radiance_and_fwhm(filepath):
    """
    提取 PRISMA 数据中的辐射率数据立方体、波段波长数组和 FWHM 数据。

    :param filepath: PRISMA HDF5 文件的路径
    :return: 包含三个元素的元组 (radiance_cube, wavelength_array, fwhm_array)
             radiance_cube 是形状为 [波段, 行, 列] 的三维数组，
             wavelength_array 是波段对应的波长数组，
             fwhm_array 是波段对应的 FWHM 数组。
    """
    try:
        with h5py.File(filepath, "r") as prisma_file:
            # 提取辐射率数据：VNIR 和 SWIR 两个部分的立方体数据
            vnir_cube = prisma_file["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube"][
                :
            ]
            swir_cube = prisma_file["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube"][
                :
            ]

            # 将 VNIR 和 SWIR 立方体组合成一个完整的立方体
            radiance_cube = np.concatenate((vnir_cube, swir_cube), axis=1)

            # 提取 VNIR 和 SWIR 对应的波长信息
            vnir_wavelengths = prisma_file[
                "/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/VNIR_Wavelength"
            ][:]
            swir_wavelengths = prisma_file[
                "/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/SWIR_Wavelength"
            ][:]

            # 提取 VNIR 和 SWIR 对应的 FWHM 信息
            vnir_fwhm = prisma_file[
                "/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/VNIR_FWHM"
            ][:]
            swir_fwhm = prisma_file[
                "/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/SWIR_FWHM"
            ][:]

            # 将波长和 FWHM 数组分别组合成完整的数组
            wavelength_array = np.concatenate((vnir_wavelengths, swir_wavelengths))
            fwhm_array = np.concatenate((vnir_fwhm, swir_fwhm))

        return radiance_cube, wavelength_array, fwhm_array

    except Exception as e:
        print(f"读取 PRISMA 文件时发生错误: {e}")
        return None, None, None


def main():
    filepath = filename  # 替换为你的 PRISMA 数据文件路径
    radiance_cube, wavelength_array, fwhm_array = get_prisma_radiance_and_fwhm(filepath)

    if (
        radiance_cube is not None
        and wavelength_array is not None
        and fwhm_array is not None
    ):
        print("辐射率数据立方体形状:", radiance_cube.shape)
        print("波段波长数组:", wavelength_array)
        print("波段 FWHM 数组:", fwhm_array)
    else:
        print("未能成功提取 PRISMA 数据。")


def save_prisma_data_as_netcdf(bands, output_path):
    """
    This function saves the extracted PRISMA band data into a NetCDF format.

    :param bands: Dictionary containing VNIR and SWIR band data and wavelengths.
    :param output_path: Path to save the NetCDF file.
    """
    try:
        # Create a dataset for VNIR and SWIR bands
        vnir_data = bands["VNIR"]["data"]
        swir_data = bands["SWIR"]["data"]

        # Create coordinates for bands
        vnir_wavelength = bands["VNIR"]["wavelength"]
        swir_wavelength = bands["SWIR"]["wavelength"]

        # Creating xarray Dataset
        data_vars = {
            "VNIR": (["band", "row", "col"], vnir_data),
            "SWIR": (["band", "row", "col"], swir_data),
        }
        coords = {
            "band_vnir": (["band_vnir"], vnir_wavelength),
            "band_swir": (["band_swir"], swir_wavelength),
            "row": np.arange(vnir_data.shape[1]),
            "col": np.arange(vnir_data.shape[2]),
        }

        # Create an xarray Dataset
        out_xr = xr.Dataset(data_vars=data_vars, coords=coords)

        # Adding CRS (coordinate reference system) metadata
        out_xr.rio.write_crs("EPSG:4326", inplace=True)

        # Save to NetCDF file
        out_xr.to_netcdf(output_path)

    except Exception as e:
        print(f"An error occurred while saving PRISMA data to NetCDF: {e}")


if __name__ == "__main__":
    with h5py.File(filename, "r") as prisma_file:
        # 提取辐射率数据：VNIR 和 SWIR 两个部分的立方体数据
        vnir_cube = prisma_file["/KDP_AUX/Fwhm_Swir_Matrix"][:]
        swir_cube = prisma_file["/KDP_AUX/Cw_Swir_Matrix"][:]
    print(vnir_cube.shape)
    print(vnir_cube[0, :])
    print(vnir_cube[0, :] - vnir_cube[500, :])
    print(swir_cube.shape)
    print(swir_cube[0, :])
    print(swir_cube[0, :] - swir_cube[500, :])
