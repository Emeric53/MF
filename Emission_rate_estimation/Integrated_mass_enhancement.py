import numpy as np
import math


# 基于IME算法进行排放量的估算
def emission_estimate(plume_array, pixel_resolution, windspeed_10m, slope, intercept, enhancement_unit='ppmm'):
    # calculate the area and the length of the plume
    nan_count = np.count_nonzero(~np.isnan(plume_array))
    pixel_area = math.pow(pixel_resolution, 2)
    plume_area = nan_count * pixel_area
    plume_length = math.sqrt(plume_area)
    # get the values of the plume
    plume_values = [value for value in plume_array.flatten() if value != -9999]
    if enhancement_unit == 'ppmm':
        # convert the unit from  ppm*m to kg/ppm*m, then calculate the integrated mass enhancement
        integrated_mass_enhancement = sum(plume_values) * 0.716 * 0.000001 * pixel_area
    elif enhancement_unit == 'ppm':
        # convert the unit from  ppm*m to kg/ppm by setting 8km as the scale of troposphere,
        # then calculate the integrated mass enhancement
        integrated_mass_enhancement = sum(plume_values) * 0.716 * 0.000001 * pixel_area * 8000
    else:
        print("The unit of the enhancement is not supported, please enter 'ppmm' or 'ppm'.")
    # calculate the effective windspeed with the formula
    effective_windspeed = slope * windspeed_10m + intercept
    # calculate the emission rate of the plume in the unit of kg/h
    emission_rate = (effective_windspeed * 3600 * integrated_mass_enhancement) / plume_length
    return emission_rate


