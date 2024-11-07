import re


# 山西省的经纬度范围
shanxi_longitude_min = 111.0  # 最小经度
shanxi_longitude_max = 114.5  # 最大经度
shanxi_latitude_min = 34.5  # 最小纬度
shanxi_latitude_max = 40.8  # 最大纬度


def is_within_province(data_name, lon_min, lon_max, lat_min, lat_max):
    match = re.search(r"E([+-]?\d+\.\d+).*N([+-]?\d+\.\d+)", data_name)
    if match:
        longitude = float(match.group(1))
        latitude = float(match.group(2))
        # 判断经纬度是否在指定省份范围内
        if lon_min <= longitude <= lon_max and lat_min <= latitude <= lat_max:
            return True
    return False


# 批量处理多个文件名
data_names = [
    "GF5B_AHSI_E115.2_N29.3_20230928_010941_L10000397512_SW",
    "GF5B_AHSI_E118.0_N30.5_20230928_010941_L10000397512_SW",
    # 这里可以加更多的文件名
]
