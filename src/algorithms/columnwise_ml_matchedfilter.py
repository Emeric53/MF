import numpy as np

import sys

sys.path.append("C:\\Users\\RS\\VSCode\\matchedfiltermethod\\src")
from algorithms.ml_matchedfilter import ml_matched_filter as mlmf


def columnwise_ml_matched_filter(
    data_cube,
    unit_absorption_spectrum,
    satellitetype,
    albedoadjust,
    iterate,
    sparsity=False,
    group_size=5,
):
    # Initialize the concentration array, matching satellite data dimensions
    bands, rows, cols = data_cube.shape
    concentration = np.full((rows, cols), np.nan)  # Set default to NaN

    if bands != len(unit_absorption_spectrum):
        raise ValueError(
            "The number of bands in the data cube must match the length of the unit absorption spectrum."
        )

    # Calculate the number of groups
    num_groups = cols // group_size
    remaining_cols = cols % group_size

    # Process each group of columns independently
    for group_idx in range(num_groups):
        # Define the range of columns in this group
        col_start = group_idx * group_size
        col_end = col_start + group_size
        columns_in_group = range(col_start, col_end)

        # Process these columns together
        current_group = data_cube[:, :, columns_in_group]
        concentration[:, col_start:col_end] = mlmf(
            current_group,
            unit_absorption_spectrum,
            satellitetype,
            iterate,
            albedoadjust,
            sparsity,
        )

    # Handle the remaining columns that don't form a complete group
    if remaining_cols > 0:
        col_start = num_groups * group_size
        columns_in_group = range(col_start, col_start + remaining_cols)
        current_group = data_cube[:, :, columns_in_group]
        concentration[:, col_start:] = mlmf(
            current_group,
            unit_absorption_spectrum,
            satellitetype,
            iterate,
            albedoadjust,
            sparsity,
        )

    return concentration
