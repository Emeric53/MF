# %%
import numpy as np
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression

import sys

sys.path.append("/home/emeric/Documents/GitHub/MF")
from utils.satellites_data import general_functions as gf

import time
import os

# built a lookup table for radiance spectrum at different circumstances
# including different methane enhancement, different sensor height,
# different sensor viewing angle, different sensor azimuth angle, different solar zenith angle,
# different surface altitude,

# after built the lookup table, we can use the lookup table to get the radiance spectrum
# and then use the radiance spectrum to generate unit absorption spectrum at that
# circumstance, and then use the unit absorption spectrum to proceed the matched filter method


def generate_radiance_lut_for_satellite(satellitename: str):
    # Define parameter ranges
    methane_range = np.arange(0, 50500, 500)  # Methane concentration enhancement (ppm)
    sza_range = np.arange(0, 95, 5)  # Solar zenith angle (degrees)
    surface_altitude_range = np.arange(0, 6, 1)  # Surface

    # Initialize an empty dictionary to store the radiance spectra

    # Function to simulate radiance spectrum using MODTRAN (this is just a placeholder for actual MODTRAN calls)
    channels_path = f"/home/emeric/Documents/GitHub/MF/data/satellite_channels/{satellitename}_channels.npz"
    if not os.path.exists(channels_path):
        print("The satellite name is wrong")
        return None
    output_file = f"/home/emeric/Documents/GitHub/MF/data/lookuptables/{satellitename}_radiance_lookup_table.npz"

    def get_simulated_radiance(methane, altitude, sza):
        filename = rf"C:\PcModWin5\Bin\batch_result\batch\{int(methane)}_{int(sza)}_{int(altitude)}_tape7.txt"
        bands, radiance_spectrum = gf.get_simulated_satellite_radiance(
            filename, channels_path, 1500, 2500
        )
        return bands, radiance_spectrum

    def buildup_lut(output_file):
        radiance_lookup_table = {}
        for methane in methane_range:
            for altitude in surface_altitude_range:
                for sza in sza_range:
                    # Simulate radiance spectrum
                    bands, simulated_radiance = get_simulated_radiance(
                        methane, altitude, sza
                    )

                    # Store the result in the lookup table with a tuple of parameters as the key
                    radiance_lookup_table[(methane, sza, altitude)] = simulated_radiance

        np.savez(
            output_file,
            wavelengths=bands,
            parameters=list(radiance_lookup_table.keys()),
            spectrum=list(radiance_lookup_table.values()),
        )

        return radiance_lookup_table

    radiance_lookup_table = buildup_lut(output_file)

    return radiance_lookup_table


# ! 基于 甲烷浓度 SZA 地面高程 查询辐射光谱
def load_satellite_radiance_lookup_table(satellitename: str):
    """
    Load the lookup table from a file.

    :param filename: Path to the file from which the lookup table will be loaded
    :return: Tuple of wavelengths and the lookup table (dictionary of enhancements and spectra)
    """
    filename = f"/home/emeric/Documents/GitHub/MF/data/lookuptables/{satellitename}_radiance_lookup_table.npz"
    data = np.load(filename)
    wavelengths = data["wavelengths"]
    parameters = data["parameters"]
    spectrum = data["spectrum"]
    lookup_table = {
        tuple(parameter): spectrum for parameter, spectrum in zip(parameters, spectrum)
    }
    return wavelengths, lookup_table


# Interpolation function to estimate radiance for non-exact parameter values
def get_satellite_radiance_spectrum_from_lut(
    satellitename: str, methane: float, sza: float, altitude: float, low, high
):
    wvls, lookup_table = load_satellite_radiance_lookup_table(satellitename)
    # Extract parameter points and corresponding radiance spectra
    points = np.array(list(lookup_table.keys()))
    values = np.array(list(lookup_table.values()))

    # Query point
    query_point = np.array([methane, sza, altitude])

    # Perform interpolation
    interpolated_radiance = griddata(points, values, query_point, method="linear")[0, :]
    condition = np.logical_and(wvls >= low, wvls <= high)
    wvls = wvls[condition]
    interpolated_radiance = interpolated_radiance[condition]
    return wvls, interpolated_radiance


# 批量化查询
def batch_get_radiance_from_lut(
    satellite_name, enhancement_range, sza, altitude, low=1500, high=2500
):
    # 加载查找表
    wavelengths, lookup_table = load_satellite_radiance_lookup_table(satellite_name)

    # 构建查询点 (甲烷增强, sza, 高度)
    query_points = np.array([[enh, sza, altitude] for enh in enhancement_range])

    # 批量插值查询
    radiance_list = griddata(
        list(lookup_table.keys()),
        list(lookup_table.values()),
        query_points,
        method="linear",
    )
    condition = np.logical_and(wavelengths >= low, wavelengths <= high)
    wavelengths = wavelengths[condition]
    radiance_list = np.array(radiance_list)[:, condition]
    return wavelengths, radiance_list


# 基于头尾浓度增强值，生成甲烷浓度增强范围
def generate_series_with_multiples_of_500(
    start_value: float, end_value: float
) -> np.ndarray:
    """
    Generate a numpy array starting and ending with the closest multiples of 500
    to start_value and end_value, and with 500 as the step size.

    :param start_value: The starting value
    :param end_value: The ending value
    :return: A numpy array with the series of 500 multiples
    """
    # Ensure start_value is less than or equal to end_value
    if start_value > end_value:
        start_value, end_value = end_value, start_value

    # Find the closest multiple of 500 greater than or equal to start_value
    first_500_multiple = np.ceil(start_value / 500) * 500

    # Find the closest multiple of 500 less than or equal to end_value
    last_500_multiple = np.floor(end_value / 500) * 500

    # Generate the multiples of 500 between first_500_multiple and last_500_multiple
    multiples_of_500 = np.arange(first_500_multiple, last_500_multiple + 1, 500)

    return multiples_of_500


# def generate_satellite_uas_for_specific_range_from_lut(
#     satellite_name: str,
#     start_enhancement: float,
#     end_enhancement: float,
#     lower_wavelength: float,
#     upper_wavelength: float,
#     sza: float,
#     altitude: float,
# ):
#     # 1. 构建enhancement的范围
#     if start_enhancement < 0:
#         start_enhancement = 0
#     if end_enhancement > 50000:
#         end_enhancement == 50000
#     enhancement_range = generate_series_with_multiples_of_500(
#         start_enhancement, end_enhancement
#     )

#     # 2. 查询所有甲烷增强值的光谱（批量处理）
#     wavelengths, radiance_list = batch_get_radiance_from_lut(
#         satellite_name, enhancement_range, sza, altitude
#     )

#     # 3. 过滤波长范围
#     condition = np.logical_and(
#         wavelengths >= lower_wavelength, wavelengths <= upper_wavelength
#     )
#     used_wavelengths = wavelengths[condition]

#     # # 假设 radiance_list 已经通过批量处理获得了光谱数据
#     # radiance_spectrum_base = radiance_list[0, condition]  # 第一个光谱作为基底
#     # radiance_spectrum_final = radiance_list[
#     #     -1, condition
#     # ]  # 最后一个光谱作为增强后的光谱

#     # # 计算透射率谱
#     # transmission_spectrum = (radiance_spectrum_final) / radiance_spectrum_base

#     total_radiance = np.log(radiance_list[:, condition])
#     total_radiance = total_radiance

#     # 4. 使用多项式拟合计算斜率（矢量化处理）
#     slopelist = np.polyfit(enhancement_range, total_radiance, deg=1)[0]

#     # # 使得拟合结果经过第一个点 (x0, 0)，我们将 x0 作为原点
#     # x0 = enhancement_range[0]

#     # # 进行线性回归，要求拟合线通过 (x0, 0)，我们不考虑截距
#     # X = enhancement_range.reshape(-1, 1) - x0  # 特征矩阵

#     # # 创建线性回归对象，并强制截距为 0
#     # model = LinearRegression(fit_intercept=False)

#     # # 结果数组，存储每个维度的斜率
#     # slopes = np.zeros(total_radiance.shape[1])  # 假设 total_radiance 有多个维度（列）

#     # # 对每个维度进行拟合（这里的维度是 total_radiance 的列）
#     # for i in range(total_radiance.shape[1]):
#     #     # 取出第 i 列作为目标值
#     #     y = total_radiance[:, i] - total_radiance[0, i]

#     #     # 进行线性回归拟合
#     #     model.fit(X, y)

#     #     # 获取拟合的斜率
#     #     slopes[i] = model.coef_[0]

#     # slopelist = []
#     # model_no_intercept = LinearRegression(fit_intercept=False)
#     # enhancement_range = (enhancement_range - np.min(enhancement_range)).reshape(-1, 1)
#     # for i in range(total_radiance.shape[1]):
#     #     total_radiance[:, i] = -(total_radiance[:, i] - np.min(total_radiance[:, i]))
#     #     model_no_intercept.fit(enhancement_range, total_radiance[:, i])
#     #     slopelist.append(model_no_intercept.coef_[0])
#     return (used_wavelengths, slopelist)


# 优化后的生成函数
def generate_satellite_uas_for_specific_range_from_lut(
    satellite_name: str,
    start_enhancement: float,
    end_enhancement: float,
    lower_wavelength: float,
    upper_wavelength: float,
    sza: float,
    altitude: float,
):
    # 1. 构建enhancement的范围
    start_enhancement = max(start_enhancement, 0)
    end_enhancement = min(end_enhancement, 50000)

    enhancement_range = generate_series_with_multiples_of_500(
        start_enhancement, end_enhancement
    )

    # 2. 查询所有甲烷增强值的光谱（批量处理）
    wavelengths, radiance_list = batch_get_radiance_from_lut(
        satellite_name, enhancement_range, sza, altitude
    )

    # 3. 过滤波长范围
    condition = np.logical_and(
        wavelengths >= lower_wavelength, wavelengths <= upper_wavelength
    )
    used_wavelengths = wavelengths[condition]

    # 4. 批量计算 log(radiance)
    total_radiance = np.log(radiance_list[:, condition])  # 直接进行批量操作

    # 5. 批量计算斜率
    # 优化：使用线性回归批量计算斜率
    model = LinearRegression(fit_intercept=False)
    enhancement_range_reshaped = enhancement_range.reshape(
        -1, 1
    )  # 将enhancement_range重塑为列向量

    # 为每一列（每一个光谱）进行拟合
    slopes = np.array(
        [
            model.fit(enhancement_range_reshaped, total_radiance[:, i]).coef_[0]
            for i in range(total_radiance.shape[1])
        ]
    )

    return used_wavelengths, slopes


def generate_transmittance_for_specific_range_from_lut(
    satellite_name: str,
    start_enhancement: float,
    end_enhancement: float,
    lower_wavelength: float,
    upper_wavelength: float,
    sza: float,
    altitude: float,
):
    # 1. 构建enhancement的范围
    if start_enhancement < 0:
        start_enhancement = 0
    if end_enhancement > 50000:
        end_enhancement == 50000

    # 2. 查询所有甲烷增强值的光谱（批量处理）
    wavelengths, radiance_list = batch_get_radiance_from_lut(
        satellite_name, np.array([start_enhancement, end_enhancement]), sza, altitude
    )

    # 3. 过滤波长范围
    condition = np.logical_and(
        wavelengths >= lower_wavelength, wavelengths <= upper_wavelength
    )
    used_wavelengths = wavelengths[condition]

    # 假设 radiance_list 已经通过批量处理获得了光谱数据
    radiance_spectrum_base = radiance_list[0, condition]  # 第一个光谱作为基底
    radiance_spectrum_final = radiance_list[
        -1, condition
    ]  # 最后一个光谱作为增强后的光谱

    # 计算透射率谱
    transmission_spectrum = (radiance_spectrum_final) / radiance_spectrum_base

    return (used_wavelengths, transmission_spectrum)


# %%
if __name__ == "__main__":
    # print("Start generating radiance lookup table")
    # generate_radiance_lut_for_satellite("AHSI")
    # generate_radiance_lut_for_satellite("EnMAP")
    # generate_radiance_lut_for_satellite("EMIT")
    # generate_radiance_lut_for_satellite("PRISMA")
    # generate_radiance_lut_for_satellite("ZY1")
    start = time.time()
    wvls, slope, slope2 = generate_satellite_uas_for_specific_range_from_lut(
        "AHSI", 10000, 20000, 2150, 2500, 50, 0
    )

    end = time.time()
    print(f"Time cost: {end - start}")
    import matplotlib.pyplot as plt

    plt.figure()
    uas = np.array(slope)
    plt.plot(wvls, slope, label=" uas1 ")
    uas = np.array(slope2)
    plt.plot(wvls, slope2, label=" uas2 ")

    plt.legend()
    plt.show()
    # wvls, slope = generate_satellite_uas_for_specific_range_from_lut(
    #     "AHSI",
    #     start_enhancement=0,
    #     end_enhancement=5000,
    #     lower_wavelength=2150,
    #     upper_wavelength=2500,
    #     sza=60,
    #     altitude=0,
    # )
    # wvls, slope1 = generate_satellite_uas_for_specific_range_from_lut(
    #     "AHSI",
    #     start_enhancement=0,
    #     end_enhancement=25000,
    #     lower_wavelength=2150,
    #     upper_wavelength=2500,
    #     sza=60,
    #     altitude=0,
    # )
    # wvls, slope2 = generate_satellite_uas_for_specific_range_from_lut(
    #     "AHSI",
    #     start_enhancement=0,
    #     end_enhancement=50000,
    #     lower_wavelength=2150,
    #     upper_wavelength=2500,
    #     sza=60,
    #     altitude=0,
    # )

# %%
