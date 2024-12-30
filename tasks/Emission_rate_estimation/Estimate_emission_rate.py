import numpy as np
from osgeo import gdal
import os
import geopandas as gpd
import rasterio
from shapely.geometry import Point

from utils.emission_estimate import Integrated_mass_enhancement as ime

# def wind_speed(plume_name: str) -> float:
#     """
#     this function is used to calculate the wind speed of the nearest point to the plume

#     need to prove: add the uncertainty of the wind speed

#     :param plume_name: str
#     :return: wind speed: float
#     """

#     plume_path = os.path.join("I:\甲烷烟羽产品\EMIT_Plumes_China_shp", plume_name)
#     filename = os.path.basename(plume_path)
#     date = filename.split("_")[4][:8]
#     plume = gpd.read_file(plume_path)
#     wind_directory = "H:\ERA5_shp"
#     wind_file_path = os.path.join(wind_directory, f"0.25_ERA5_wind_{date}.shp")
#     wind_data = gpd.read_file(wind_file_path)
#     min_distance = float("inf")
#     nearest_wind_point = None
#     # 遍历每个风点
#     for index, wind_point in wind_data.iterrows():
#         # 计算当前风点到烟羽面的最近距离
#         distance = plume.geometry.distance(wind_point.geometry).min()
#         # 检查是否是到目前为止找到的最近距离
#         if distance < min_distance:
#             min_distance = distance
#             nearest_wind_point = wind_point

#     # if nearest_wind_point is not None:
#     #     print(f"Nearest wind speed: {nearest_wind_point['wind_speed']} m/s")
#     #     print(f"Nearest wind direction: {nearest_wind_point['wind_direc']} degrees")
#     # else:
#     #     print("No wind points found.")
#     return nearest_wind_point["wind_speed"]


def wind_speed(plume_name: str) -> float:
    """
    This function is used to calculate the wind speed of the nearest point to the plume,
    and then expand to a 3x3 grid around it to compute the mean and standard deviation of wind speed.

    :param plume_name: str
    :return: Tuple: (mean_wind_speed: float, std_dev: float)
    """

    # Read the plume shapefile
    plume_path = os.path.join("J:\甲烷烟羽产品\EMIT_Plumes_China_shp", plume_name)
    # filename = os.path.basename(plume_path)
    # date = filename.split("_")[4][:8]
    date = "20230217"
    plume = gpd.read_file(plume_name)

    # Read the ERA5 wind shapefile
    wind_directory = "L:\ERA5_shp"
    wind_file_path = os.path.join(wind_directory, f"0.25_ERA5_wind_{date}.shp")
    wind_data = gpd.read_file(wind_file_path)

    # Initialize the minimum distance and the nearest wind point
    min_distance = float("inf")
    nearest_wind_point = None

    # Iterate through each wind point to find the nearest point to the plume
    for index, wind_point in wind_data.iterrows():
        distance = plume.geometry.distance(wind_point.geometry).min()
        if distance < min_distance:
            min_distance = distance
            nearest_wind_point = wind_point

    # Get the coordinates of the nearest wind point
    nearest_wind_point_coords = nearest_wind_point.geometry.coords[0]

    # Define the grid for the 3x3 area around the nearest point
    x_min = nearest_wind_point_coords[0] - 0.25  # 0.25 as an example grid size
    x_max = nearest_wind_point_coords[0] + 0.25
    y_min = nearest_wind_point_coords[1] - 0.25
    y_max = nearest_wind_point_coords[1] + 0.25

    # Select the wind data points within the 3x3 grid area
    grid_points = wind_data.cx[x_min:x_max, y_min:y_max]

    # Calculate the mean and standard deviation of wind speeds within the grid
    wind_speeds = grid_points[
        "wind_speed"
    ].values  # Assuming "wind_speed" is the column name
    mean_wind_speed = np.mean(wind_speeds)
    std_dev = np.std(wind_speeds)

    return mean_wind_speed, std_dev


def wind_speed_from_raster(plume_tif: str, wind_shp: str):
    """
    This function calculates the wind speed mean and standard deviation from the 3x3 pixel grid
    around the plume center in a raster file.

    :param plume_name: str, plume shapefile path (if needed for further information)
    :param wind_raster_path: str, wind speed raster file path
    :return: tuple of (mean_wind_speed, std_wind_speed)
    """

    # 读取烟羽的.tif文件（rasterio）
    with rasterio.open(plume_tif) as plume_src:
        plume_transform = plume_src.transform  # 获取烟羽栅格的空间变换信息
        plume_width = plume_src.width
        plume_height = plume_src.height

    # 获取烟羽的几何中心（假设烟羽为矩形区域）
    plume_center_x, plume_center_y = plume_transform * (
        plume_width / 2,
        plume_height / 2,
    )

    # 读取风速的.shp文件（geopandas）
    wind_data = gpd.read_file(wind_shp)

    # 计算每个风速点到烟羽中心的距离
    wind_data["distance"] = wind_data.geometry.distance(
        Point(plume_center_x, plume_center_y)
    )

    # 按照距离升序排序，选择最近的9个风速点（3x3）
    nearest_points = wind_data.nsmallest(9, "distance")

    # 获取这9个风速点的风速值（假设有名为 'wind_speed' 的列）
    wind_speeds = nearest_points[
        "wind_speed"
    ].values  # 这里假设风速列的名字是 'wind_speed'

    # 计算风速的均值和标准差
    mean_wind_speed = np.mean(wind_speeds)
    std_wind_speed = np.std(wind_speeds)

    return mean_wind_speed, std_wind_speed


def read_tif_array(tif_path) -> np.ndarray:
    # read the array of the plume
    data = gdal.Open(tif_path, gdal.GA_ReadOnly)
    return data.ReadAsArray()


def read_tif_with_nodata(tif_path: str):
    # 打开.tif文件
    with rasterio.open(tif_path) as src:
        # 读取数据
        data = src.read(1)  # 读取第一个波段的数据

        # 获取NoData值
        nodata_value = src.nodata

        if nodata_value is not None:
            # 将NoData值替换为np.nan
            data = np.where(data == nodata_value, np.nan, data)

    return data


from scipy.spatial import cKDTree
import numpy as np


def compute_effective_length_using_sqrt(date):
    """
    计算所有有效值像素个数的平方根，作为有效烟羽的长度
    """
    # 提取有效点（大于0的部分）
    valid_pixels = np.sum(data > 0)  # noqa: E999

    # 计算有效长度（有效像素个数的平方根）
    effective_length = np.sqrt(valid_pixels)

    return effective_length


def compute_effective_length_from_pixel_count(data, pixel_resolution):
    """
    计算所有有效值像素个数的平方根，作为有效烟羽的长度，考虑空间分辨率
    """
    # 提取有效点（大于0的部分）
    valid_pixels = np.sum(data > 0)

    # 计算有效长度（有效像素个数的平方根，单位为像素）
    effective_length_pixels = np.sqrt(valid_pixels)

    # 乘以像素分辨率，得到实际的有效长度（单位为米）
    effective_length = effective_length_pixels * pixel_resolution
    return effective_length


def compute_effective_length_with_pca(data, pixel_resolution):
    """
    使用PCA计算烟羽的有效长度（基于烟羽的有效点），考虑空间分辨率
    """
    # 提取有效点（大于0的部分）
    valid_points = extract_valid_points(data)

    # 使用PCA进行主轴分析
    pca = PCA(n_components=2)
    pca.fit(valid_points)

    # 获取主轴的长度（单位为像素）
    axis_lengths = pca.singular_values_

    # 乘以空间分辨率，得到实际的有效长度（单位为米）
    effective_length = axis_lengths[0] * pixel_resolution  # 主轴方向的长度
    return effective_length


def compute_effective_length_using_bounding_box(data, pixel_resolution):
    """
    使用最小外接矩形法计算烟羽的有效长度，考虑空间分辨率
    """
    # 提取有效点（大于0的部分）
    valid_points = extract_valid_points(data)

    # 将有效点转换为轮廓形式
    contour = valid_points.reshape((-1, 1, 2))

    # 计算最小外接矩形
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 计算矩形的长轴（主轴）作为有效长度（单位为像素）
    width = np.linalg.norm(box[0] - box[1])
    length = np.linalg.norm(box[1] - box[2])

    # 乘以空间分辨率，得到实际的有效长度（单位为米）
    effective_length = max(width, length) * pixel_resolution
    return effective_length


def calculate_distances_with_kdtree(data, pixel_resolution):
    """
    计算烟羽中两个有效像素之间的最大和最小距离，作为有效长度的估算。

    :param data: 输入的烟羽数据，包含NaN值和有效的像素
    :param pixel_resolution: 每个像素的实际地面长度（单位为米）
    :return: 最大和最小距离（单位为米）
    """
    # 获取所有非NaN像素的坐标
    non_nan_indices = np.array(np.where(~np.isnan(data)))
    points = np.column_stack(non_nan_indices.T)  # 合并为 (x, y) 坐标对

    if len(points) < 2:
        return 0, 0  # 如果有效点小于两个，返回0

    # 创建 KDTree
    tree = cKDTree(points)

    # 查询所有点之间的距离
    distances, _ = tree.query(points, k=2)  # k=2 查找每个点的最近邻

    # 获取最大和最小的距离（单位为像素）
    max_distance = np.max(distances)
    min_distance = np.min(distances)

    # 转换为实际的地面距离（单位为米）
    max_distance_meters = max_distance * pixel_resolution
    min_distance_meters = min_distance * pixel_resolution

    return max_distance_meters, min_distance_meters


def extract_valid_points(data):
    """
    提取有效的烟羽点（大于0的部分）
    """
    # 获取所有有效点的坐标（大于0的部分）
    valid_points = np.argwhere(data > 0)
    return valid_points


# # set the filepath of methane plume image or the enhancement of methane
# plume_folder = r"C:\Users\RS\Desktop"
# plume_names = ["plume.tif"]

# # 批量读取烟羽tiff文件
# for plume_name in plume_names:
#     # 读取 烟羽的浓度数据
#     plume_filepath = os.path.join(plume_folder, plume_name)
#     plume_data = read_tif_with_nodata(plume_filepath)
#     # 获取 预估的10m风速
#     wind_shp_path = r"L:\ERA5_shp\0.25_ERA5_wind_20230217.shp"
#     windspeed, std = wind_speed_from_raster(plume_filepath, wind_shp_path)
#     print(plume_filepath + "   Wind speed: " + str(windspeed) + " m/s")
#     print(plume_filepath + "   Wind speed std: " + str(std) + " m/s")
#     # 基于 IME算法进行排放量估算
#     print(np.nansum(plume_data))
#     emission_rate_result, result_std = ime.emission_estimate(
#         plume_data,
#         pixel_resolution=60,
#         windspeed_10m=windspeed,
#         wind_speed_10m_std=std,
#         slope=0.38,
#         intercept=0.41,
#         enhancement_unit="ppmm",
#     )
#     print(plume_filepath + "   Emission rate: " + str(emission_rate_result) + " kg/h")
#     print(plume_filepath + "   Emission rate std: " + str(result_std) + " kg/h")
#     print("---------------------------------------------------------------")


# effective_length_sqrt = compute_effective_length_from_pixel_count(plume_data, 30)
# print(f"Effective length (sqrt): {effective_length_sqrt}")

# effective_length_mutual = calculate_distances_with_kdtree(plume_data, 30)
# print(f"Max distance: {effective_length_mutual[0]}")


# # 计算有效长度（使用PCA）
# effective_length_pca = compute_effective_length_with_pca(plume_data, 30)
# print(f"有效长度（PCA）：{effective_length_pca}")

# # 计算有效长度（使用最小外接矩形）
# effective_length_bounding_box = compute_effective_length_using_bounding_box(
#     plume_data, 30
# )
# print(f"有效长度（最小外接矩形）：{effective_length_bounding_box}")

#     max, min = calculate_distances_with_kdtree(plume_data)
#     print(plume_filepath + "   Max distance: " + str(max))
#     print(plume_filepath + "   Min distance: " + str(min))
# 获取 预估的10m风速


plume_folder = r"C:\Users\RS\Desktop"
plume_names = ["plumed_concentratioin1.tif"]
plume_filepath = "G:\EMIT_plume_result_1sigma\EMIT_L1B_RAD_001_20230327T073331_2308605_002_plumeproduct.tif"  # for plume_name in plume_names:
#     # 读取 烟羽的浓度数据
#     plume_filepath = os.path.join(plume_folder, plume_name)

plume_filepath = r"/media/emeric/Glass/EMITL2BCH4PLM_001-20241108_114914/shanxi/EMIT_L2B_CH4PLM_001_20230418T060625_000824.tif"

plume_data = read_tif_with_nodata(plume_filepath)

wind_shp_path = r"/media/emeric/Documents/ERA5_shp/0.25_ERA5_wind_20230418.shp"
windspeed, windspeed_std = wind_speed_from_raster(plume_filepath, wind_shp_path)
print(plume_filepath + "   Wind speed: " + str(windspeed) + " m/s")
print(plume_filepath + "   Wind speed std: " + str(windspeed_std) + " m/s")

# 基于 IME算法进行排放量估算
print(np.nansum(plume_data))

emission_rate_result, result_std, plume_area, plume_length = ime.emission_estimate(
    plume_data,
    pixel_resolution=30,
    windspeed_10m=windspeed,
    wind_speed_10m_std=windspeed_std,
    slope=0.38,
    intercept=0.41,
    enhancement_unit="ppmm",
)
print("plume_area:", plume_area)
print("plume_length:", plume_length)
print(plume_filepath + "   Emission rate: " + str(emission_rate_result) + " kg/h")
print(plume_filepath + "   Emission rate std: " + str(result_std) + " kg/h")
print("---------------------------------------------------------------")
