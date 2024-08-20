import numpy as np
import xarray as xr
import os


# 读取emit的数组
def get_emit_array(file_path: str) -> np.array:
    """
    Reads a nc file and returns a NumPy array containing all the bands.

    :param file_path: the path of the nc file
    :return: a 3D NumPy array with shape (bands, height, width)
    """
    try:
        dataset = xr.open_dataset(file_path)
        # 读取EMIT radiance的数据，并将其转置为 波段 行 列 的维度形式
        radiance_array = dataset['radiance'].values.transpose(2, 0, 1)

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
        dataset = xr.open_dataset(file_path, group='sensor_band_parameters')
        # 读取EMIT radiance的数据，并将其转置为 波段 行 列 的维度形式
        bands_array = dataset['wavelengths'].values

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
    return bands[indices], data[indices,:,:]


# 将结果导出为nc文件
def export_array_to_nc(data_array: np.array, input_nc_path: str, output_folder: str, original_data: np.array):
    """
    Export a NumPy array to a NetCDF (.nc) file, maintaining the geo-referencing and metadata
    from the input NetCDF file, and adding the original uncorrected data.

    :param data_array: NumPy array containing the processed data to be exported
    :param input_nc_path: Path to the input NetCDF file (used for referencing and metadata)
    :param output_folder: Directory to save the output NetCDF file
    :param original_data: NumPy array containing the original, uncorrected data
    """
    try:
        filename = os.path.basename(input_nc_path).replace('.nc', '_enhanced.nc')
        output_path = os.path.join(output_folder, filename)

        root_ds = xr.open_dataset(input_nc_path)
        location_ds = xr.open_dataset(input_nc_path, group='location')

        GLT_NODATA_VALUE = 0
        glt_array = np.nan_to_num(np.stack([location_ds['glt_x'].data, location_ds['glt_y'].data], axis=-1),
                                  nan=GLT_NODATA_VALUE).astype(int)

        fill_value = -9999
        out_ds = np.full((glt_array.shape[0], glt_array.shape[1]), fill_value, dtype=np.float32)
        valid_glt = np.all(glt_array != GLT_NODATA_VALUE, axis=-1)
        glt_array[valid_glt] -= 1
        out_ds[valid_glt] = data_array[glt_array[valid_glt, 1], glt_array[valid_glt, 0]]

        GT = root_ds.geotransform
        dim_x = location_ds.glt_x.shape[1]
        dim_y = location_ds.glt_x.shape[0]
        lon = np.zeros(dim_x)
        lat = np.zeros(dim_y)

        for x in range(dim_x):
            x_geo = GT[0] + (x + 0.5) * GT[1]
            lon[x] = x_geo
        for y in range(dim_y):
            y_geo = GT[3] + (y + 0.5) * GT[5]
            lat[y] = y_geo

        coords = {'lat': (['lat'], lat), 'lon': (['lon'], lon)}
        data_vars = {'methane_enhancement': (['lat', 'lon'], out_ds)}

        out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=root_ds.attrs)
        out_xr['methane_enhancement'].attrs = {'Content': 'Methane enhancement in the atmosphere', 'Units': 'ppm·m'}
        out_xr.coords['lat'].attrs = location_ds['lat'].attrs
        out_xr.coords['lon'].attrs = location_ds['lon'].attrs
        # 保存初始的 NextCDF 文件
        out_xr.rio.write_crs(root_ds.spatial_ref, inplace=True)
        out_xr.to_netcdf(output_path)
        out_xr.close()
        
        data_vars = {'methane_enhancement': (['crosstrack', 'alongtrack'], original_data)}
        out_xr = xr.Dataset(data_vars=data_vars, attrs=root_ds.attrs)
        out_xr.to_netcdf(output_path.replace("_enhanced.nc", "_original_enhanced.nc"))
        out_xr.close()

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except IOError as io_error:
        print(f"I/O error: {io_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    filepath = "I:\\EMIT\\Radiance_data\\EMIT_L1B_RAD_001_20220810T064957_2222205_033.nc"
    emit_bands = get_emit_bands(filepath)
    print(emit_bands)

if __name__ == '__main__':
    main()
