# get the mask of the plume from the methane enhancement image
import numpy
from osgeo import gdal
import numpy as np
import scipy.stats as stats
import xarray as xr


# get the path of the methane enhancement images
def get_raster_array(filepath):
    # 利用gdal打开数据
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    return dataset

def flood_fill(data,mask, x, y):
    """Fills a connected region on a screen with a new color.

    Args:
        screen: A 2D list of pixel colors.
        x: The starting x-coordinate.
        y: The starting y-coordinate.
        new_color: The color to fill the region with.
    """

    value = methane[x][y]
    # set a threshold as the background concentration
    if value >= 200:
        data[x][y] = 1
filepath = ''
ds = xr.open_dataset(filepath)

source_coor = [5, 7]
methane = ds['methane_enhancement'].to_numpy()
cols, rows = methane.shape

mask = numpy.zeros([cols, rows])






