import numpy as np
import xarray as xr
import os
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point


# 读取emit的数组
def get_emit_array(file_path: str) -> np.ndarray:
    """
    Reads a nc file and returns a NumPy array containing all the bands.

    :param file_path: the path of the nc file
    :return: a 3D NumPy array with shape (bands, height, width)
    """
    try:
        dataset = xr.open_dataset(file_path)
        # 读取EMIT radiance的数据，并将其转置为 波段 行 列 的维度形式
        radiance_array = dataset["radiance"].values.transpose(2, 0, 1)

        if dataset is None:
            raise FileNotFoundError(f"Unable to open file: {file_path}")

    except Exception as e:
        print(f"Error: {e}")
        return None
    return radiance_array


# 从数据文件中 获取 emit的通道波长信息
def get_emit_bands(file_path: str) -> np.ndarray:
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


# 直接读取先前保存的波长信息
def read_emit_bands():
    """
    get bands list of ahsi
    :param band_file:  filepath containing bands wavelength
    :return: bands list
    """
    # 读取校准文件
    wavelengths = np.load(
        "/home/emeric/Documents/GitHub/MF/data/satellite_channels/EMIT_channels.npz"
    )["central_wvls"]
    return wavelengths


# 获取 筛选范围后的波长数组和radiance信息
def get_emit_bands_array(
    file_path: str, bot: float, top: float
) -> tuple[np.ndarray, np.ndarray]:
    bands = read_emit_bands()
    data = get_emit_array(file_path=file_path)
    indices = np.where((bands >= bot) & (bands <= top))[0]
    return bands[indices], data[indices, :, :]


# 基于nc文件读取当前影像的 SZA 和 高程
def get_sza_altitude(filepath):
    try:
        obspath = filepath.replace("_RAD_", "_OBS_")
        dataset = xr.open_dataset(obspath)
        # 读取EMIT 几何参数的数组
        bands_array = dataset["obs"].values
        sza = np.average(bands_array[:, :, 4])
        if dataset is None:
            raise FileNotFoundError(f"Unable to open file: {filepath}")

    except Exception as e:
        print(f"Error: {e}")
        return None
    return sza, 0


# 判断该影像是否位于指定的区域[经纬度范围]
def is_within_region(filepath, region):
    lon_min, lon_max, lat_min, lat_max = region[:]
    try:
        obspath = filepath.replace("_RAD_", "_OBS_")
        dataset = xr.open_dataset(obspath, group="location")
        if dataset is None:
            raise FileNotFoundError(f"Unable to open file: {filepath}")
        # 读取EMIT 几何参数的数组
        lat = dataset["lat"].values
        lon = dataset["lon"].values
        longitude = np.average(lon)
        latitude = np.average(lat)
    except Exception as e:
        print(f"Error: {e}")
        return False
    if lon_min <= longitude <= lon_max and lat_min <= latitude <= lat_max:
        return True
    return False


# 判断该影像是否位于指定的区域[shapefile]
def is_within_region_shapefile(filepath, province_region_shapefile):
    try:
        obspath = filepath.replace("_RAD_", "_OBS_")
        dataset = xr.open_dataset(obspath, group="location")
        if dataset is None:
            raise FileNotFoundError(f"Unable to open file: {filepath}")
        # 读取EMIT 几何参数的数组
        lat = dataset["lat"].values
        lon = dataset["lon"].values
        longitude = np.average(lon)
        latitude = np.average(lat)
    except Exception as e:
        print(f"Error: {e}")
        return False
    point = Point(longitude, latitude)
    if province_region_shapefile.contains(point).any():
        return True
    return False


# 将结果导出为GeoTIFF文件
def export_to_geotiff(out_xr, output_tiff_path):
    try:
        # 获取数据和坐标
        data_array = out_xr["methane_enhancement"].values  # 提取数据
        lon_array = out_xr.coords["lon"].values  # 提取经度
        lat_array = out_xr.coords["lat"].values  # 提取纬度

        # 获取地理变换信息
        lon_min, lon_max = lon_array.min(), lon_array.max()
        lat_min, lat_max = lat_array.min(), lat_array.max()

        # 根据原始数据创建像素的分辨率，假设经纬度范围已知
        pixel_size_x = (lon_max - lon_min) / len(lon_array)  # 经度方向的像素分辨率
        pixel_size_y = (lat_max - lat_min) / len(lat_array)  # 纬度方向的像素分辨率

        # 创建地理变换对象
        transform = from_origin(lon_min, lat_max, pixel_size_x, pixel_size_y)

        # 创建 GeoTIFF 文件
        with rasterio.open(
            output_tiff_path,
            "w",
            driver="GTiff",
            count=1,
            dtype="float32",
            width=len(lon_array),
            height=len(lat_array),
            crs="EPSG:4326",
            transform=transform,
            nodata=-9999,
        ) as dst:
            dst.write(data_array, 1)

        print(f"GeoTIFF file saved to {output_tiff_path}")

    except Exception as e:
        print(f"An error occurred while exporting to GeoTIFF: {e}")


# 将结果导出为tiff文件
def export_emit_array_to_tif(
    data_array: np.ndarray,
    input_nc_path: str,
    output_path: str,
):
    """
    Export a NumPy array to a NetCDF (.nc) file, maintaining the geo-referencing and metadata
    from the input NetCDF file, and adding the original uncorrected data.

    :param data_array: NumPy array containing the processed data to be exported
    :param input_nc_path: Path to the input NetCDF file (used for referencing and metadata)
    :param output_folder: Directory to save the output NetCDF file
    :param original_data: NumPy array containing the original, uncorrected data
    """
    try:
        root_ds = xr.open_dataset(input_nc_path)
        location_ds = xr.open_dataset(input_nc_path, group="location")

        GLT_NODATA_VALUE = 0
        glt_array = np.nan_to_num(
            np.stack([location_ds["glt_x"].data, location_ds["glt_y"].data], axis=-1),
            nan=GLT_NODATA_VALUE,
        ).astype(int)

        fill_value = -9999
        out_ds = np.full(
            (glt_array.shape[0], glt_array.shape[1]), fill_value, dtype=np.float32
        )
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

        coords = {"lat": (["lat"], lat), "lon": (["lon"], lon)}
        data_vars = {"methane_enhancement": (["lat", "lon"], out_ds)}

        out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=root_ds.attrs)
        out_xr["methane_enhancement"].attrs = {
            "Content": "Methane enhancement in the atmosphere",
            "Units": "ppm·m",
        }
        out_xr.coords["lat"].attrs = location_ds["lat"].attrs
        out_xr.coords["lon"].attrs = location_ds["lon"].attrs

        export_to_geotiff(out_xr, output_path)

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except IOError as io_error:
        print(f"I/O error: {io_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# 将结果导出为nc文件 和 tiff文件
def export_emit_array_to_nc_tif(
    data_array: np.ndarray,
    input_nc_path: str,
    output_folder: str,
):
    """
    Export a NumPy array to a NetCDF (.nc) file, maintaining the geo-referencing and metadata
    from the input NetCDF file, and adding the original uncorrected data.

    :param data_array: NumPy array containing the processed data to be exported
    :param input_nc_path: Path to the input NetCDF file (used for referencing and metadata)
    :param output_folder: Directory to save the output NetCDF file
    :param original_data: NumPy array containing the original, uncorrected data
    """
    try:
        filename = os.path.basename(input_nc_path).replace(".nc", "_enhanced.nc")
        output_path = os.path.join(output_folder, filename)

        root_ds = xr.open_dataset(input_nc_path)
        location_ds = xr.open_dataset(input_nc_path, group="location")

        GLT_NODATA_VALUE = 0
        glt_array = np.nan_to_num(
            np.stack([location_ds["glt_x"].data, location_ds["glt_y"].data], axis=-1),
            nan=GLT_NODATA_VALUE,
        ).astype(int)

        fill_value = -9999
        out_ds = np.full(
            (glt_array.shape[0], glt_array.shape[1]), fill_value, dtype=np.float32
        )
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

        coords = {"lat": (["lat"], lat), "lon": (["lon"], lon)}
        data_vars = {"methane_enhancement": (["lat", "lon"], out_ds)}

        out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=root_ds.attrs)
        out_xr["methane_enhancement"].attrs = {
            "Content": "Methane enhancement in the atmosphere",
            "Units": "ppm·m",
        }
        out_xr.coords["lat"].attrs = location_ds["lat"].attrs
        out_xr.coords["lon"].attrs = location_ds["lon"].attrs

        export_to_geotiff(out_xr, output_path.replace(".nc", ".tif"))
        out_xr.rio.write_crs(root_ds.spatial_ref, inplace=True)
        out_xr.to_netcdf(output_path)
        out_xr.close()

    except FileNotFoundError as fnf_error:
        print(f"File not found: {fnf_error}")
    except IOError as io_error:
        print(f"I/O error: {io_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# 将原始nc文件的rgb波段导出为tiff文件
def export_emit_rgb_array_to_tif(filepath, outputfolder):
    radaince_cube = get_emit_array(filepath)
    # 获取 rgb 真彩色
    r = radaince_cube[35, :, :]  # 36
    g = radaince_cube[21, :, :]  # 22
    b = radaince_cube[10, :, :]  # 10
    rgb = np.stack((r, g, b), axis=0)
    outputpath = os.path.join(
        outputfolder, os.path.basename(filepath).replace(".nc", "_RGB.tif")
    )

    root_ds = xr.open_dataset(filepath)
    location_ds = xr.open_dataset(filepath, group="location")

    GLT_NODATA_VALUE = 0
    glt_array = np.nan_to_num(
        np.stack([location_ds["glt_x"].data, location_ds["glt_y"].data], axis=-1),
        nan=GLT_NODATA_VALUE,
    ).astype(int)

    fill_value = -9999
    out_ds = np.full(
        (3, glt_array.shape[0], glt_array.shape[1]), fill_value, dtype=np.float32
    )
    valid_glt = np.all(glt_array != GLT_NODATA_VALUE, axis=-1)
    glt_array[valid_glt] -= 1
    for band in range(3):
        out_ds[band, valid_glt] = rgb[
            band, glt_array[valid_glt, 1], glt_array[valid_glt, 0]
        ]

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

    coords = {"lat": (["lat"], lat), "lon": (["lon"], lon)}
    data_vars = {"methane_enhancement": (["band", "lat", "lon"], out_ds)}

    out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=root_ds.attrs)
    out_xr["methane_enhancement"].attrs = {
        "Content": "Methane enhancement in the atmosphere",
        "Units": "ppm·m",
    }
    out_xr.coords["lat"].attrs = location_ds["lat"].attrs
    out_xr.coords["lon"].attrs = location_ds["lon"].attrs

    try:
        # 获取数据和坐标
        data_array = out_xr["methane_enhancement"].values  # 提取数据 (3, height, width)
        lon_array = out_xr.coords["lon"].values  # 提取经度
        lat_array = out_xr.coords["lat"].values  # 提取纬度

        # 获取地理变换信息
        lon_min, lon_max = lon_array.min(), lon_array.max()
        lat_min, lat_max = lat_array.min(), lat_array.max()

        # 计算像素分辨率
        pixel_size_x = (lon_max - lon_min) / data_array.shape[2]  # 经度方向的像素分辨率
        pixel_size_y = (lat_max - lat_min) / data_array.shape[1]  # 纬度方向的像素分辨率

        # 创建地理变换对象
        transform = from_origin(lon_min, lat_max, pixel_size_x, pixel_size_y)

        # 创建多波段 GeoTIFF 文件
        with rasterio.open(
            outputpath,
            "w",
            driver="GTiff",
            count=3,  # 三个波段
            dtype="float32",
            width=data_array.shape[2],  # 列数
            height=data_array.shape[1],  # 行数
            crs="EPSG:4326",
            transform=transform,
            nodata=-9999,
        ) as dst:
            # 写入每个波段
            for band in range(3):
                dst.write(data_array[band, :, :], band + 1)

        print(f"GeoTIFF file with 3 bands saved to {outputpath}")

    except Exception as e:
        print(f"An error occurred while exporting to GeoTIFF: {e}")


if __name__ == "__main__":
    pass
