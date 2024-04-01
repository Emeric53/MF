import os

import cdsapi

c = cdsapi.Client()

### specify the index of selected region

ia = 1

## SKIP Certain Arae or Year
downloadedAreas = ['', ]
downloadedYears = ['', ]

## index:  0      1      2     3     4     5     6     7
Regions = ['ASIA', 'EUAF', 'AFR', 'NAM', 'SAM', 'MLY', 'AUS', 'NZD', ]  # 'EBor','WBor','antarctic','arctic']

Areas = [
    [89, -178.5, -89, 178.5],  ### world grib  [89,-178.5,-89,178.5]?
    [60, 70, 0, 145, ],  ### ASIA
    [60, -20, 0, 70, ],  ## EUAF
    [0, 7, -36, 52, ],  ## AFR
    [60, -140, 15, -50, ],  ## NAM
    [15, -95, -56, -34, ],  ## SAM
    [0, 97, -11, 163, ],  ### MLY
    [-11, 113, -44, 155, ],  ### AUS
    [-34, 166, -48, 179, ],  ### NZD
    # [75, 0, 60, 180,], ### EBor
    # [75, -180, 60, -15,], ### WBor
    # [-56, -180, -90, 180,], ### Antarctic
    # [75, -180, 90, 180,], ### Arctic
]

''' general setups '''
Years = ['2019', '2020']
Months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
Days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
        '19', '20',
        '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
utc_times = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
             '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
             '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
             '18:00', '19:00', '20:00', '21:00', '22:00', '23:00', ]
leapyears = ['2012', '2016', '2020', ]

# 29 Layers
plevels = ['50', '70', '100', '125', '150', '175', '200', '225', '250', '300', '350', '400', '450', '500', '550',
           '600', '650', '700', '750', '775', '800', '825', '850', '875', '900', '925', '950', '975', '1000', ]

params = ['temperature', 'relative_humidity']  # 'relative_humidity', 'temperature']

resolution = ['1', '1']

''''''
## loops for downloading
area = Areas[ia]
UTC = utc_times
for i in range(len(site_lon)):
    for iy, year in enumerate(Years):
        for im, month in enumerate(Months):
            for id, day in enumerate(Days):
                if month in ['04', '06', '09', '11'] and day >= '31':
                    continue
                if year in leapyears and month == '02' and day >= '30':
                    continue
                if (year not in leapyears) and month == '02' and day >= '29':
                    continue
                ncFileName = "C:/Users/RS/Desktop/ERA5/" + 'ERA5_' + year + month + day+ '_site'+ str(i+1) + '.nc'
                if os.path.exists(ncFileName):
                    continue
                print(ncFileName + " is downloading")
                c.retrieve('reanalysis-era5-pressure-levels',  # 'reanalysis-era5-pressure-levels',
                           {'product_type': 'reanalysis',
                            'variable': params,
                            'pressure_level': plevels,
                            'year': year,
                            'month': month,
                            'day': day,
                            'time': UTC,
                            'format': 'netcdf',
                            'area': [site_lat[i]+0.5,site_lon[i],site_lat[i]-0.5,site_lon[i]+0.5],
                            'grid': resolution,
                            },
                           ncFileName)
