import os
import geopandas as gpd
import pandas as pd
from osgeo import gdal
from shapely.geometry import box


def create_shapefile_from_tiffs(directory,output_path):
    # 存储每个TIFF的bounding box的列表
    bounding_boxes = []

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith("plume.tif") or filename.endswith("plume.tiff"):
            filepath = os.path.join(directory, filename)
            dataset = gdal.Open(filepath)

            if dataset is None:
                continue

            # 获取地理变换参数
            transform = dataset.GetGeoTransform()
            x_min = transform[0]
            y_max = transform[3]
            x_max = x_min + transform[1] * dataset.RasterXSize
            y_min = y_max + transform[5] * dataset.RasterYSize

            # 创建一个矩形polygon
            bbox = box(x_min, y_min, x_max, y_max)
            bounding_boxes.append(gpd.GeoDataFrame({'TIFF_Name': [filename]},index=[0], crs=dataset.GetProjection(), geometry=[bbox]))

    # 合并所有GeoDataFrame到一个文件中
    if bounding_boxes:
        all_bboxes = gpd.GeoDataFrame(pd.concat(bounding_boxes, ignore_index=True))
        all_bboxes.to_file(output_path)


# 输入文件夹和输出文件路径
# input_folder = "I:\\甲烷烟羽产品\\CarnonMapper_all_plumes\\export_2016-01-01_2017-01-01"
# output_path = "I:\\甲烷烟羽产品\\CarnonMapper_all_plumes\\carbonmapper_2016.shp"
input_folder = "I:\甲烷烟羽产品\EMIT_Plumes_China"
output_path = "I:\甲烷烟羽产品\EMIT_Plumes_China_shp\EMIT_plumes_China.shp"
create_shapefile_from_tiffs(input_folder, output_path)
