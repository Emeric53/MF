from osgeo import gdal
# get the mask of the plume from the methane enhancement image


# get the path of the methane enhancement images
def get_raster_array(filepath):
    # 利用gdal打开数据
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    return dataset.ReadAsArray()







