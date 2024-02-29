import cdsapi

# 创建CDS API客户端
c = cdsapi.Client()

# 指定下载参数
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', # 分别为东西向和南北向的风速分量
        ],
        'year': '2023',
        'month': '02',
        'day': '21',
        'time': '12:00',
        'grid': [0.02, 0.02],
        'area': [44.15, 87.75, 43.95, 87.95], # 北纬53°-18°, 东经73°-135°，覆盖中国大部分地区
        'format': 'netcdf', # 选择NetCDF格式
        },
    "C:\\Users\\RS\\Desktop\\ERA5\\p1_20230221.nc") # 指定下载文件的名称

# 定义下载参数
product_type = "reanalysis"
variable = "10m_u_component_of_wind"
year = "2023"
month = "01"
day = "01"
area = [50, -10, 70, 30]  # 北纬50度，西经10度，北纬70度，东经30度

# 创建下载请求列表
requests = []
for year in range(2020, 2024):
    for month in range(1, 13):
        requests.append(
            cdsapi.Request(
                product_type=product_type,
                variable=variable,
                year=str(year),
                month=str(month),
                day=str(day),
                area=area,
                format="grib",
            )
        )

# 批量下载数据
cdsapi.download_multiple_requests(requests)

# 打印下载的文件名
for request in requests:
    print(request.filename)


