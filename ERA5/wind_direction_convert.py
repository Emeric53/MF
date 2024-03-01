import xarray as xr
import numpy as np
from osgeo import osr
from osgeo import gdal, ogr
import geopandas as gpd
from shapely.geometry import Point
import metpy.calc as mpcalc
import pathlib as pl

filefolder = pl.Path("C:\\Users\\RS\\Desktop\\ERA5")
filepaths = filefolder.glob('*.nc')


def calculate_wind_speed_dir(u, v):
    speed = np.sqrt(u ** 2 + v ** 2)
    direction = (180 + (180 / np.pi) * np.arctan2(u, v)) % 360  # Meteorological convention
    return speed, direction

for filepath in filepaths:
    ds = xr.open_dataset(filepath)
    u10 = ds['u10'].values[0, :, :]
    v10 = ds['v10'].values[0, :, :]
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
    outputpath = filefolder / (filepath.name.rstrip('.nc') + '.shp')
    gdf.to_file(outputpath)

