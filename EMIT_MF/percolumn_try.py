import geopandas as gpd
import xarray as xr
import rasterio
import numpy as np



def nc_to_tiff(nc_file_path, output_tif_path,  resolution=None):
    """Converts a NetCDF file to a GeoTIFF.

    Args:
        nc_file_path (str): Path to the input NetCDF file.
        output_tif_path (str): Path to the output GeoTIFF file.
        variable_name (str): Name of the variable in the NetCDF file to convert.
        resolution (tuple, optional): Desired output resolution in degrees
                                      (x resolution, y resolution). Defaults to None,
                                      in which case it's calculated from the data.
    """

    # Load NetCDF file
    ds = xr.open_dataset(nc_file_path)
    data_array = ds["radiance"].values
    data_array = data_array.transpose(2,0,1)[0,:,:]
    # Extract geospatial information
    loc = xr.open_dataset(nc_file_path, group='location')
    lons = loc['lon'].values
    lats = loc['lat'].values

    # CRS handling
    try:
        crs = ds.crs.crs_wkt
    except AttributeError:
        crs = 'epsg:4326'  # Default to WGS84 if CRS absent

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        data_array.flatten(),
        geometry=gpd.points_from_xy(lons.flatten(), lats.flatten()),
        crs=crs
    )
    # Determine raster extent and resolution
    xmin, ymin, xmax, ymax = gdf.total_bounds
    if resolution is None:
        # Calculate a reasonable resolution if none is provided
        resolution = (
            (xmax - xmin) / data_array.shape[1],
            (ymax - ymin) / data_array.shape[0]
        )

    # Create rasterio transform
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, *resolution)

    # Create output GeoTIFF
    with rasterio.open(
            output_tif_path,
            'w',
            driver='GTiff',
            height=int((ymax - ymin) / resolution[1]),
            width=int((xmax - xmin) / resolution[0]),
            count=1,
            dtype=data_array.dtype,
            crs=crs,
            transform=transform,
            nodata=np.nan  # Optional - set a nodata value
    ) as dst:
        dst.write(gdf.interpolate("nearest").values, 1)


# Example usage
nc_file_path = "F:\\EMIT_DATA\\nc\\EMIT_L1B_RAD_001_20220810T064957_2222205_033.nc"
output_tif_path = "C:\\Users\\RS\\Desktop\\Plume_images\\test.tif"

nc_to_tiff(nc_file_path, output_tif_path)



