"""
   this code is used to process the radiance file by using the matching filter algorithm
   and the goal is to get the methane enhancement image
"""

# the necessary lib to be imported
import os
import numpy as np
import pathlib as pl
import array as xr
from osgeo import gdal


# a function to get the raster array and return the dataset
def get_raster_array(filepath):
    # open the dataset from the filepath
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    return dataset


# define a function to get the dataarray from the nc file
def read_nc_to_array(filepath):
    dataset = xr.open_dataset(filepath)
    radiance = dataset['radiance'].values
    return radiance


# a function to open the unit absorption spectrum file and return the numpy array
def open_unit_absorption_spectrum(filepath, min, max):
    # open the unit absorption spectrum file and convert it a numpy array
    with open(filepath, 'r') as file:
        data = file.readlines()
        uas_list = [float(line.split(" ")[1].rstrip('\n')) for line in data if min <= float(line.split(' ')[0]) <= max]
    out_put = np.array(uas_list)
    return out_put


# define the main function to process the radiance file by using the matching filter algorithm
def mf_process(filepath, uas_path, output_path, is_iterate=False, is_albedo=False, is_filter=False):
    # open the unit absorption spectrum file and convert it a numpy array
    unit_absorption_spectrum = open_unit_absorption_spectrum(uas_path,2100,2500)

    # get the raster array from the radiance file
    radiance_data = np.array(read_nc_to_array(str(filepath)))
    radiance_window = radiance_data[:, :, radiance_data.shape[2]-len(unit_absorption_spectrum) -1 :-1]

    name = pl.Path(filepath).name.rstrip(".nc")
    # pre-define the list to store the band data and the count of the non-nan value


    # get the number of bands, rows and columns of the image data
    rows, cols, bands = radiance_window.shape
    # build the albedo, alpha to store the needing result
    u = np.zeros(bands)
    target = np.zeros(bands)

    albedo = np.zeros((rows, cols))
    alpha = np.zeros((rows, cols))
    # calculate the covariance matrix
    c = np.zeros((bands, bands))

    # alone the row axis to get the mean value of each band for each column
    for col_index in range(cols):
        # get the data of each column
        column_data = radiance_window[:, col_index, :]
        # count the nan number in the column data
        count_not_nan = np.count_nonzero(~np.isnan(column_data))
        if count_not_nan == 0:
            alpha[:, col_index] = np.nan
            continue
        # get the mean value of each band for each column
        u = np.nanmean(column_data, axis=0)
        # based on the background spectrum and the unit absorption spectrum, calculate the target spectrum
        target = np.multiply(u, unit_absorption_spectrum)

        # alone the row axis to calculate the covariance matrix
        for row in range(rows):
            if not np.isnan(column_data[row, 0]):
                c += np.outer(column_data[row, :] - u, column_data[row, :] - u)

        # get the average of the sum of covariance matrix
        c = c / count_not_nan

        # get the inverse of the covariance matrix
        c_inverse = np.linalg.inv(c)

        # iterate the whole region of the image data to calculate the albedo and the alpha for each pixel
        for row_index in range(rows):
            # account the nan value in the image data and make the result to be nan
            if not np.isnan(column_data[row_index, 0]):
                if is_albedo:
                    # based on the formula to calculate the albedo of each pixel and store it into the albedo array
                    albedo[row_index, col_index] = np.inner(column_data[row_index, :], u) / np.inner(u, u)
                else:
                    albedo[row_index, col_index] = 1
                # based on the formula to calculate the methane enhancement of each pixel
                # and store it in the alpha array
                up = (column_data[row_index, :] - u) @ c_inverse @ target
                down = albedo[row_index, col_index] * (target @ c_inverse @ target)
                alpha[row_index, col_index] = up / down
            else:
                alpha[row_index, col_index] = np.nan

        if is_iterate:
            # build the l1_filter to store the result
            l1filter = np.zeros((rows, cols))
            # define a tiny value to avoid the zero division
            epsilon = np.finfo(np.float32).tiny
            # based on the former alpha result, update the background spectrum and the target spectrum
            # and then update the covariance matrix
            # and get the new methane enhancement result
            iter_data = column_data.copy()
            for iter_num in range(4):
                for row_index in range(rows):
                    if not np.isnan(column_data[row_index, 0]):
                        iter_data[row_index, :] = column_data[row_index, :] - albedo[row_index, col_index] * target * alpha[row_index, col_index]
                        if is_filter:
                            l1filter[row_index, col_index] = 1 / (alpha[row_index, col_index] + epsilon)
                        else:
                            l1filter[row_index, col_index] = 0

                count_not_nan = np.count_nonzero(~np.isnan(iter_data[:, 0]))
                u = np.nanmean(iter_data, axis=0)
                target = np.multiply(u, unit_absorption_spectrum)
                # get the new covariance matrix
                c = c*0
                for row_index in range(rows):
                    if not np.isnan(iter_data[row_index, 0]):
                        c += np.outer(iter_data[row_index, :] - u, iter_data[row_index, :] - u)
                c = c / count_not_nan
                # get the inverse of the covariance matrix
                c_inverse = np.linalg.inv(c)

                # calculate the new methane enhancement result
                for row_index in range(rows):
                        if not np.isnan(column_data[row_index, 0]):
                            up = (column_data[row_index, :] - u) @ c_inverse @ target - l1filter[row_index, col_index]
                            down = albedo[row_index, col_index] * target @ c_inverse @ target
                            alpha[row_index, col_index] = max(up / down, 0)
                        else:
                            alpha[row_index, col_index] = np.nan

    # set the output tiff file path
    #  get the number of rows and columns of the alpha array
    output_folder = str(output_path)

    # use the function to export the methane enhancement result to a nc file
    export_result_to_netcdf_or_tiff(alpha, filepath, output_folder)

        # maybe directly convert into tiff file


# define the function to export the methane enhancement result to a nc file
def export_result_to_netcdf_or_tiff(ds_array, filepath, output_folder):
    ds_array


# define the path of the unit absorption spectrum file and open it
uas_filepath = 'New_ppm_m_EMIT_unit_absorption_spectrum.txt'
uas = open_unit_absorption_spectrum(uas_filepath, 1500, 2500)
print(uas)

# based on the code to decide process mode
process_mode = 2

if process_mode == 0:
    
    # run in batch:
    # define the path of the radiance folder and get the radiance file list with an img suffix
    radiance_folder = "I:\\EMIT\\rad"
    radiance_path_list = pl.Path(radiance_folder).glob('*.nc')

    # get the output file path and get the existing output file list to avoid the repeat process
    root = pl.Path("I:\\EMIT\\methane_result\\Direct_result")
    output = root.glob('*.nc')
    outputfile = []
    for i in output:
        outputfile.append(str(i.name))

    # the input includes the radiance file path, the unit absorption spectrum, the output path and the is_iterate flag
    for radiance_path in radiance_path_list:
        current_filename = str(radiance_path.name)
        if current_filename in outputfile:
            continue
        else:
            print(f"{current_filename} is now being processed")
            try:
                mf_process(radiance_path, uas_path=uas_filepath, output_path=root, is_iterate=True, is_albedo=True, is_filter=True)
                print(f"{current_filename} has been processed")
            except Exception as e:
                print(f"{current_filename} has an error")
                pass
            
            
elif process_mode == 1:
    
    # run a single file
    file_path = pl.Path(r"I:\\EMIT\\rad\\EMIT_L1B_RAD_001_20230204T041009_2303503_016.nc")
    # EMIT_L2B_CH4PLM_001_20220818T070105_000508.tif
    # EMIT_L2B_CH4PLM_001_20230217T063221_000646.tif
    # EMIT_L2B_CH4PLM_001_20230221T045604_000716.tif
    # EMIT_L2B_CH4PLM_001_20230420T060148_000837.tif
    # EMIT_L2B_CH4PLM_001_20230424T042444_000856.tif
    # EMIT_L2B_CH4PLM_001_20230609T045106_000937.tif
    # EMIT_L2B_CH4PLM_001_20230627T030822_000060.tif
    # EMIT_L2B_CH4PLM_001_20230805T060827_000999.tif
    single_result_output = pl.Path("I:\\EMIT\\methane_result\\plume_onebyone")
    filename = os.path.basename(file_path)
    print(f"{file_path} is now being processed.")
    try:
        mf_process(file_path, uas_path=uas_filepath, output_path=single_result_output, is_iterate=False, is_albedo=True, is_filter=True)
        print(f"{filename} has been processed")
    except Exception as e:
        print(e)
        print(f"{filename} has an error")
