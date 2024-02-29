import cdsapi
import calendar
from subprocess import call

if __name__ == '__main__':
    c = cdsapi.Client()  # 创建用户
    # 数据信息字典
    dic = {
        'product_type': 'reanalysis',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', # 分别为东西向和南北向的风速分量
        ],
        'year': '',
        'month': '',
        'day': '',
        'time': ['12:00', '15:00', '18:00', '21:00'],# 选择12:00, 15:00, 18:00, 21:00时次的数据
        'grid': [0.02, 0.02],
        'area': [],#北纬53°-18°,东经73°-135°，覆盖中国大部分地区
        'format': 'netcdf',#选择NetCDF格式
        }
    point = ["point3", "point4", "point5", "point6"]
    coor = [[39.2,109.05], [39.08,109.41], [38.92,111.09], [40.55,117.67]]
    plumes = {
    "point3": [
        {"name": "plume3", "data": ['2023', "06",'29']}
    ],
    "point4": [
        {"name": "plume3", "data": ['2022', "08",'20']},
        {"name": "plume4", "data": ['2023', "03",'26']},
    ],
    "point5": [
        {"name": "plume3", "data": ['2023', "02",'04']}
    ],
    "point6": [
        {"name": "plume3", "data": ['2023', "03",'27']}
    ]}

    # 通过循环批量下载1979年到2020年所有月份数据
    for index in range(len(point)):  # 遍历年
        for plumeindex in range(len(plumes[point[index]])):  # 遍历月
            # 将年、月、日更新至字典中
            dic['year'] = plumes[point[index]][plumeindex]['data'][0]
            dic['month'] = plumes[point[index]][plumeindex]['data'][1]
            dic['day'] = plumes[point[index]][plumeindex]['data'][2]
            dic['area'] = [coor[index][0] + 0.2, coor[index][1] - 0.2, coor[index][0] - 0.2, coor[index][1] + 0.2]
            path = "C:\\Users\\RS\\Desktop\\ERA5\\" +point[index] + "_" + dic['year'] + dic['month']+dic['day']+'.nc'  # 文件名
            r = c.retrieve('reanalysis-era5-single-levels', dic, path)
