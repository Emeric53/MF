import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import pathlib as pl

filefolder = pl.Path("H:\\ERA5_Data")
filepaths = filefolder.glob('*.nc')
outputfolder = pl.Path("H:\\ERA5_shp")

def calculate_wind_speed_dir(u, v):
    """
    Calculate wind speed and direction from its u and v components.

    Parameters:
    u (array-like): The u-component (east-west) of the wind vector.
    v (array-like): The v-component (north-south) of the wind vector.

    Returns:
    tuple: A tuple containing two elements:
        - speed (array-like): The calculated wind speed.
        - direction (array-like): The calculated wind direction in degrees (meteorological convention).
    
    The wind direction is calculated using the meteorological convention where 0° indicates wind coming from the north, 90° from the east, 180° from the south, and 270° from the west.
    """
    speed = np.sqrt(u ** 2 + v ** 2)
    direction = (180 + (180 / np.pi) * np.arctan2(u, v)) % 360  # Meteorological convention
    return speed, direction

for filepath in filepaths:
    ds = xr.open_dataset(filepath)
    u10 = np.array(ds['u10'].values)
    v10 = np.array(ds['v10'].values)
    # u10 = np.average(u10, axis=0)
    # v10 = np.average(v10, axis=0)
    u10 = u10[1, :, :]
    v10 = v10[1, :, :]
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    wind_speed, wind_direction = calculate_wind_speed_dir(u10, v10)
    data = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            lon = lons[j]
            lat = lats[i]
            point = Point(lon, lat)
            data.append([point, wind_speed[i, j], wind_direction[i,j]])
    gdf = gpd.GeoDataFrame(data, columns=['geometry', 'wind_speed', 'wind_direction'])
    outputpath = outputfolder / (filepath.name.rstrip('.nc') + '.shp')
    gdf.to_file(outputpath)

