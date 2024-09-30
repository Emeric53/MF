import numpy as np
import xarray as xr
import h5py as he5


filename = (
    "I:\\PRISMA_控制释放实验\\PRS_L1_STD_OFFL_20221015181614_20221015181618_0001.he5"
)


# 打开 HE5 文件
with he5.File(filename, "r") as f:
    # 定义一个函数来打印每个数据集的维度信息
    def print_dataset_info(name, obj):
        if isinstance(obj, he5.Dataset):
            print(f"数据集名称: {name}")
            print(f"数据维度: {obj.shape}")
            print(f"数据类型: {obj.dtype}")
            print("-" * 30)

    # 递归访问所有数据集并打印信息
    f.visititems(print_dataset_info)


# 读取emit的数组
def get_prisma_array(file_path: str) -> np.array:
    """
    Reads a nc file and returns a NumPy array containing all the bands.

    :param file_path: the path of the nc file
    :return: a 3D NumPy array with shape (bands, height, width)
    """
    try:
        dataset = he5.read_he5(file_path)
        dataset = xr.open_dataset(file_path)
        # 读取EMIT radiance的数据，并将其转置为 波段 行 列 的维度形式
        radiance_array = dataset["radiance"].values.transpose(2, 0, 1)

        if dataset is None:
            raise FileNotFoundError(f"Unable to open file: {file_path}")

    except Exception as e:
        print(f"Error: {e}")
        return None
    return radiance_array


# 获取 emit的通道波长信息
def get_emit_bands(file_path: str) -> np.array:
    """
    Reads a nc file and returns a NumPy array containing all the bands.

    :param file_path: the path of the nc file
    :return: a 3D NumPy array with shape (bands, height, width)
    """
    try:
        dataset = xr.open_dataset(file_path, group="sensor_band_parameters")
        # 读取EMIT radiance的数据，并将其转置为 波段 行 列 的维度形式
        bands_array = dataset["wavelengths"].values

        if dataset is None:
            raise FileNotFoundError(f"Unable to open file: {file_path}")

    except Exception as e:
        print(f"Error: {e}")
        return None
    return bands_array


# 获取 筛选范围后的波长数组和radiance信息
def get_emit_bands_array(file_path, bot, top):
    bands = get_emit_bands(file_path=file_path)
    data = get_emit_array(file_path=file_path)
    indices = np.where((bands >= bot) & (bands <= top))[0]
    return bands[indices], data[indices, :, :]


def main():
    filepath = (
        "I:\\EMIT\\Radiance_data\\EMIT_L1B_RAD_001_20220810T064957_2222205_033.nc"
    )
    emit_bands = get_emit_bands(filepath)
    print(emit_bands)


if __name__ == "__main__":
    main()
