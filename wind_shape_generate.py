from osgeo import gdal
import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# Load the wind speed TIFF
wind_speed_raster = rasterio.open(r"C:\Users\RS\Downloads\drive-download-20240402T125220Z-001\WindSpeed_on_20211109T03.tif")
wind_speed_array = wind_speed_raster.read(1)  # Reads the first band

# Load the wind direction TIFF
wind_direction_raster = rasterio.open(r"C:\Users\RS\Downloads\drive-download-20240402T125220Z-001\WindDirection_on_20211109T03.tif")
wind_direction_array = wind_direction_raster.read(1)  # Reads the first band

print(wind_speed_array.shape)
# Generate points from the raster
points = []
for row in range(wind_speed_array.shape[0]):
    for col in range(wind_speed_array.shape[1]):
        # Get the wind speed and direction for the current cell
        speed = wind_speed_array[row, col]
        direction = wind_direction_array[row, col]

        # Get the geographic coordinates for the current cell
        x, y = wind_speed_raster.transform * (col, row)
        # Create a point geometry
        point = Point(x, y)

        # Append to list as a tuple with speed and direction
        points.append((point, speed, direction))

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(points, columns=['geometry', 'wind_speed', 'wind_direction'])

gdf.to_file("C:\\Users\\RS\\Downloads\\Wind_20211109T03.shp")


