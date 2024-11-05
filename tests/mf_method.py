import math
import numpy as np

import sys

sys.path.append("../src/")

from utils.satellites_data import general_functions as gf


def ml_matched_filter(radiance_cube, unit_absorption_spectrum):
    band_num, row_num, col_num = radiance_cube.shape
    methane_enhancement = np.zeros((row_num, col_num))

    background_radiance = np.average(radiance_cube, axis=(1, 2))
    target_spectrum = background_radiance * unit_absorption_spectrum
    d_covariance = radiance_cube - background_radiance
    covariance_matrix = np.zeros((band_num, band_num))
    for row in range(row_num):
        for col in range(col_num):
            covariance_matrix += np.dot(
                d_covariance[:, row, col], d_covariance[:, row, col]
            )
    covariance_matrix /= row_num * col_num
    covariance_inversed = np.linalg.pinv(covariance_matrix)

    for row in range(row_num):
        for col in range(col_num):
            methane_enhancement[row, col] = (
                (radiance_cube[:, row, col] - background_radiance)
                @ covariance_inversed
                @ target_spectrum
                / (target_spectrum @ covariance_inversed @ target_spectrum)
            )

    methane_enhancement_mean = np.mean(methane_enhancement)
    methane_enhancement_std = np.std(methane_enhancement)
    high_pixel_percentage = np.count_nonzero(methane_enhancement> np.ones_like(methane_enhancement)*(methane_enhancement_mean+methane_enhancement_std)) / (row_num * col_num)
    if high_pixel_percentage > 0.36 
        print("High methane enhancement detected!")
        