import numpy as np
import xarray as xr
import h5py


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
            swir_cube = prisma_file["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube"][
                :
            ]
            swir_cube = np.transpose(swir_cube, (1, 2, 0))
            swir_cube = np.flip(swir_cube, axis=0)
            swir_wavelengths = np.array(prisma_file.attrs["List_Cw_Swir"])[:-2]
            swir_fwhm = np.array(prisma_file.attrs["List_Fwhm_Swir"])[:-2]
            # Reverse the wavelengths and FWHM
            wavelengths = np.flip(swir_wavelengths)
            fwhm = np.flip(swir_fwhm)
        return swir_cube, wavelengths, fwhm

    except Exception as e:
        print(f"读取 PRISMA 文件时发生错误: {e}")
        return None, None, None


def get_prisma_array(filepath):
    try:
        with h5py.File(filepath, "r") as prisma_file:
            # 提取辐射率数据：VNIR 和 SWIR 两个部分的立方体数据
            swir_cube = prisma_file["/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube"][
                :
            ]
            swir_cube = np.transpose(swir_cube, (1, 2, 0))
            swir_cube = np.flip(swir_cube, axis=0)
            factor = prisma_file.attrs["ScaleFactor_Swir"]
            offset = prisma_file.attrs["Offset_Swir"]
            swir_cube = swir_cube / factor - offset

        return swir_cube

    except Exception as e:
        print(f"读取 PRISMA 文件时发生错误: {e}")
        return None, None, None


def get_prisma_channels():
    wavelengths = np.load(
        "C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src\\data\\satellite_channels\\Prisma_channels.npz"
    )["central_wvls"]
    return wavelengths


def get_prisma_bands_array(
    file_path: str, bot: float, top: float
) -> tuple[np.ndarray, np.ndarray]:
    bands = get_prisma_channels()
    data = get_prisma_array(file_path)
    indices = np.where((bands >= bot) & (bands <= top))[0]
    return bands[indices], data[indices, :, :]


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
    filename = "I:\stanford_campaign\PRISMA\PRS_L1_STD_OFFL_20211016183624_20211016183629_0001.he5"

    wavelength_array, radiance_cube = get_prisma_bands_array(filename, 1500, 2100)
    print(radiance_cube.shape)
    print(radiance_cube[0, :, :])
    print(wavelength_array)
    # main()
