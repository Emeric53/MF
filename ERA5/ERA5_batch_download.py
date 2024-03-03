import cdsapi
import calendar
from subprocess import call

if __name__ == '__main__':
    c = cdsapi.Client()  # 创建用户
    # 数据信息字典
    dic = {
        'product_type': 'reanalysis',
        'variable': [
            '', '10m_v_component_of_wind', # 分别为东西向和南北向的风速分量
        ],
        'year': '',
        'month': '',
        'day': '',
        'time': ['0:00','1:00','4:00', '5:00'],# 选择12:00, 15:00, 18:00, 21:00时次的数据
        'grid': [3, 2],
        'area': [],#北纬53°-18°,东经73°-135°，覆盖中国大部分地区
        'format': 'netcdf',#选择NetCDF格式
        }
    point = ["point1","point3","point5"]
    coor = [[44.05, 87.85], [39.2,109.05],  [38.92,111.09]]

    # 通过循环批量下载1979年到2020年所有月份数据
    for index in range(len(point)):  # 遍历年
        for plumeindex in range(len(plumes[point[index]])):  # 遍历月
            # 将年、月、日更新至字典中
            dic['area'] = [coor[index][0] + 0.2, coor[index][1] - 0.2, coor[index][0] - 0.2, coor[index][1] + 0.2]
            path = "C:\\Users\\RS\\Desktop\\ERA5\\" +point[index] + "_" + dic['year'] + dic['month']+dic['day']+'.nc'  # 文件名
            r = c.retrieve('reanalysis-era5-single-levels', dic, path)
