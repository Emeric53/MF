import numpy as np
from osgeo import gdal
import Integrated_mass_enhancement as ime
import os
import geopandas as gpd
import sys

sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod")
from MyFunctions import needed_functions as nf


def wind_speed(plume_name) -> float:
    """
    this function is used to calculate the wind speed of the nearest point to the plume

    need to prove: add the uncertainty of the wind speed

    :param plume_name: str
    :return: wind speed: float
    """

    plume_path = os.path.join("I:\甲烷烟羽产品\EMIT_Plumes_China_shp", plume_name)
    filename = os.path.basename(plume_path)
    date = filename.split("_")[4][:8]
    plume = gpd.read_file(plume_path)
    wind_directory = "H:\ERA5_shp"
    wind_file_path = os.path.join(wind_directory, f"0.25_ERA5_wind_{date}.shp")
    wind_data = gpd.read_file(wind_file_path)
    min_distance = float("inf")
    nearest_wind_point = None
    # 遍历每个风点
    for index, wind_point in wind_data.iterrows():
        # 计算当前风点到烟羽面的最近距离
        distance = plume.geometry.distance(wind_point.geometry).min()
        # 检查是否是到目前为止找到的最近距离
        if distance < min_distance:
            min_distance = distance
            nearest_wind_point = wind_point

    # if nearest_wind_point is not None:
    #     print(f"Nearest wind speed: {nearest_wind_point['wind_speed']} m/s")
    #     print(f"Nearest wind direction: {nearest_wind_point['wind_direc']} degrees")
    # else:
    #     print("No wind points found.")
    return nearest_wind_point["wind_speed"]


def read_tif_array(tif_path) -> np.ndarray:
    # read the array of the plume
    data = gdal.Open(tif_path, gdal.GA_ReadOnly)
    return data.ReadAsArray()


# set the filepath of methane plume image or the enhancement of methane
plume_folder = r"I:\\甲烷烟羽产品\\EMIT_Plumes_China"
plume_names = [
    "EMIT_L2B_CH4PLM_001_20230204T040649_000617.tif",
    "EMIT_L2B_CH4PLM_001_20220818T070105_000508.tif",
    "EMIT_L2B_CH4PLM_001_20230217T063221_000646.tif",
    "EMIT_L2B_CH4PLM_001_20230221T045604_000716.tif",
    "EMIT_L2B_CH4PLM_001_20230420T060148_000837.tif",
    "EMIT_L2B_CH4PLM_001_20230424T042444_000856.tif",
    "EMIT_L2B_CH4PLM_001_20230609T045106_000937.tif",
    "EMIT_L2B_CH4PLM_001_20230627T030822_000060.tif",
    "EMIT_L2B_CH4PLM_001_20230204T040649_000617.tif",
    "EMIT_L2B_CH4PLM_001_20230805T060827_000999.tif",
    "EMIT_L2B_CH4PLM_001_20230327T073331_000772.tif",
    "EMIT_L2B_CH4PLM_001_20230326T081955_000771.tif",
    "EMIT_L2B_CH4PLM_001_20220820T052804_000514.tif",
]
# 批量读取烟羽tiff文件
for plume_name in plume_names:
    # 读取 烟羽的浓度数据
    plume_filepath = os.path.join(plume_folder, plume_name)
    plume_data = nf.read_tiff(plume_filepath)
    # 获取 预估的10m风速
    windspeed = wind_speed(plume_name.replace(".tif", ".shp"))
    # 基于 IME算法进行排放量估算
    emission_rate_result = ime.emission_estimate(
        plume_data,
        pixel_resolution=30,
        windspeed_10m=windspeed,
        slope=0.38,
        intercept=0.41,
        enhancement_unit="ppmm",
    )
    print(plume_filepath + "   Emission rate: " + str(emission_rate_result) + " kg/h")
    print("---------------------------------------------------------------")
