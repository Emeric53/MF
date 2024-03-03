import cdsapi
import calendar
from subprocess import call

if __name__ == '__main__':
    c = cdsapi.Client()  # 创建用户
    first_year = 2018
    last_year = 2021
    for year in range(first_year, last_year + 1):
        for month in range(1, 13):
            print("=========================================================")
            print("Downloading {year}-{month:02d}".format(year=year, month=month))
            dic ={'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': '2m_temperature',
                    'year': str(year),
                    'month': "{month:02d}".format(month=month),
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area': [
                        90, 170, 80,
                        180,
                    ],
                    'format': 'grib',
                }}

        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '2m_temperature', 'surface_pressure',
            ],
            'year': [
                '2018', '2019', '2020',
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                90, -180, -90,
                180],
            'grid':[2, 3],
        },
        'download.nc')
    # 数据信息字典
    dic = {

        'format': 'netcdf',#选择NetCDF格式
        }
    point = ["point3",  "point5"]
    coor = [[39.2,109.05],  [38.92,111.09]]
    plumes = {
    "point3": [
        {"name": "plume3", "data": ['2023', "06",'29']}
    ],
    "point5": [
        {"name": "plume3", "data": ['2023', "02",'04']}
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
