import numpy as np
import math


# 基于IME算法进行排放量的估算
def IME_emission_estimate(
    plume_array,
    pixel_resolution,
    windspeed_10m,
    wind_speed_10m_std,
    slope=0.33,
    intercept=0.67,
    enhancement_unit="ppmm",
):
    """
    Estimate the emission rate of the plume using the Integrated Mass Enhancement algorithm.
    Args:
        plume_array (numpy.ndarray): The array of the plume.
        pixel_resolution (int): The resolution of the pixel in meters.
        windspeed_10m (float): The wind speed at 10 meters in m/s.
        wind_speed_10m_std (float): The standard deviation of the wind speed at 10 meters in m/s.
        slope (float): The slope of the regression line.
        intercept (float): The intercept of the regression line.
        enhancement_unit (str): The unit of the enhancement, either 'ppmm' or 'ppm'.
    Returns:
        tuple: A tuple containing the emission rate, the uncertainty of the emission rate, the area of the plume, and the length of the plume.
    """
    # calculate the area and the length of the plume

    IME = np.nansum(plume_array)

    nan_count = np.count_nonzero(~np.isnan(plume_array))
    pixel_area = math.pow(pixel_resolution, 2)
    plume_area = nan_count * pixel_area
    plume_length = math.sqrt(plume_area)
    print(f"plume_area: {plume_area:.2f}")
    print(f"plume_length: {plume_length:.2f}")

    # get the values of the plume
    # 转换系数还需再考虑 可能是  1 ppmm 甲烷质量为  7.15 * 10^{-4} g
    if enhancement_unit == "ppmm":
        # convert the unit from  ppm*m to kg/ppm*m, then calculate the integrated mass enhancement
        integrated_mass_enhancement = IME * 7.16 * 10e-7 * pixel_area
    elif enhancement_unit == "ppm":
        # convert the unit from  ppm*m to kg/ppm by setting 8km as the scale of troposphere,
        # then calculate the integrated mass enhancement
        integrated_mass_enhancement = sum(IME) * 0.716 * 0.000001 * pixel_area * 8000
    else:
        print(
            "The unit of the enhancement is not supported, please enter 'ppmm' or 'ppm'."
        )

    # calculate the effective windspeed with the formula
    effective_windspeed = slope * windspeed_10m + intercept
    effective_windspeed_std = slope * wind_speed_10m_std

    # calculate the emission rate of the plume in the unit of kg/h
    emission_rate = (
        effective_windspeed * 3600 * integrated_mass_enhancement
    ) / plume_length

    # calculate the uncertainty of the emission rate
    emission_rate_uncertainty = (
        effective_windspeed_std * 3600 * integrated_mass_enhancement
    ) / plume_length
    return emission_rate, emission_rate_uncertainty, plume_area, plume_length
