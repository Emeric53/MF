import numpy as np
import xarray as xr
import os


# 数据读取相关
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


# 将结果导出为nc文件
def export_array_to_nc(result: np.array, filepath: str, output_folder: str):
    """
    Export a NumPy array to a GeoTIFF file with the same geo-referencing as the input file.
    :param result: NumPy array to be exported
    :param filepath: Path to the input GeoTIFF file
    :param output_folder: Folder to save the output GeoTIFF file
    """
    try:
        filename = os.path.basename(filepath).replace('.nc', '_enhanced.nc')
        # open the root dataset and the location dataset
        ds = xr.open_dataset(filepath)
        loc = xr.open_dataset(filepath, group='location')

        # set the nodata value for glt and stack the x and y arrays together
        GLT_NODATA_VALUE = 0
        glt_array = np.nan_to_num(np.stack([loc['glt_x'].data, loc['glt_y'].data], axis=-1),
                                  nan=GLT_NODATA_VALUE).astype(int)

        # get the radiance array from the root dataset
        ds_array = result

        # Build Output Dataset
        # the fill value is set to -9999
        fill_value = -9999
        # get an array with the same shape as the glt array and fill it with the fill value -9999
        out_ds = np.zeros((glt_array.shape[0], glt_array.shape[1]), dtype=np.float32) + fill_value

        # get a boolean array with the same shape as the glt array where the values are True if the glt array is
        # not equal to the nodata value
        valid_glt = np.all(glt_array != GLT_NODATA_VALUE, axis=-1)

        # Adjust for One based Index
        # subtract 1 from the glt array where the valid_glt array is True
        glt_array[valid_glt] -= 1

        # Use indexing/broadcasting to populate array cells with 0 values
        out_ds[valid_glt] = ds_array[glt_array[valid_glt, 1], glt_array[valid_glt, 0]]

        # get the geotransform from the root dataset
        GT = ds.geotransform

        # Create Array for Lat and Lon and fill
        dim_x = loc.glt_x.shape[1]
        dim_y = loc.glt_x.shape[0]
        lon = np.zeros(dim_x)
        lat = np.zeros(dim_y)

        # fill the lat and lon arrays with the geotransform values
        for x in np.arange(dim_x):
            x_geo = (GT[0] + 0.5 * GT[1]) + x * GT[1]
            lon[x] = x_geo
        for y in np.arange(dim_y):
            y_geo = (GT[3] + 0.5 * GT[5]) + y * GT[5]
            lat[y] = y_geo

        # unpacks the existing dictionary from the wvl dataset.
        coords = {'lat': (['lat'], lat), 'lon': (['lon'], lon)}
        data_vars = {'methane_enhancement': (['lat', 'lon'], out_ds)}

        out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)
        out_xr['methane_enhancement'].attrs = {'Content': 'Methane enhancement in the atmosphere', 'Units': 'ppm·m'}
        out_xr.coords['lat'].attrs = loc['lat'].attrs
        out_xr.coords['lon'].attrs = loc['lon'].attrs
        out_xr.rio.write_crs(ds.spatial_ref, inplace=True)
        output_path = output_folder + filename
        out_xr.to_netcdf(output_path)
        print(f"File saved successfully at {output_path}")
    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except IOError as io_error:
        print(io_error)
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    filepath = "I:\\EMIT\\Radiation_data\\EMIT_L1B_RAD_001_20220810T064957_2222205_033.nc"
    emit_bands = get_emit_bands(filepath)
    print(type(emit_bands))
    np.save('emit_bands.npy', emit_bands)


if __name__ == '__main__':
    main()
