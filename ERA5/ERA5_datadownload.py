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
        'month': '01',
        'day': '01',
        'time': '12:00',
        'grid': [0.5, 0.5],
        'area': [53, 73, 18, 135], # 北纬53°-18°, 东经73°-135°，覆盖中国大部分地区
        'format': 'netcdf', # 选择NetCDF格式
        },
    "C:\\Users\\RS\\Desktop\\ERA5\\China_2023010112.nc") # 指定下载文件的名称
