""" This code is used to calculate the cross section flux of the plume.
by computing the flux through one or more plume cross sections orthogonal to the plume axis.
"""
import numpy as np
from osgeo import gdal
import math

# Set the path of the plume
plume_filepath = r"C:\Users\RS\Desktop\EMIT\MethanePlume\EMIT_L2B_CH4PLM_001_20230204T041009_000618_tiff.tif"
plume_filepath = plume_filepath.replace('\\', '/')
# Read the plume data and remove the invalid values
plume_data = gdal.Open(plume_filepath, gdal.GA_ReadOnly)
plume_data = plume_data.ReadAsArray()
# Set the pixel resolution, unit: m
pixel_resolution = 60000

# sourcerate = the intergral of the product of the plume concentration and the wind speed over the plume cross section



