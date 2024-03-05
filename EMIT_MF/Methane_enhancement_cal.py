"""
this code is used to process the radiance file by using the matching filter algorithm
and the goal is to get the methane enhancement image
"""
# the necessary lib to be imported
import numpy as np
from osgeo import gdal
import pathlib as pl
import xarray as xr

# a function to get the raster array and return the dataset
def get_raster_array(filepath):
    # open the dataset from the filepath
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    return dataset

def read_nc_to_array(filepath):
    dataset = xr.open_dataset(filepath)
    radiance = dataset['radiance'].values

    return radiance

filepath = '/Users/nomoredrama/Downloads/EMIT_L1B_RAD_001_20220814T051412_2222604_005.nc'
radiance = read_nc_to_array(filepath)

print(radiance.shape)

# a function to open the unit absorption spectrum file and return the numpy array
def open_unit_absorption_spectrum(filepath):
    # open the unit absorption spectrum file and convert it a numpy array
    uas_list = []
    with open(filepath, 'r') as file:
        data = file.readlines()
        for band in data:
            split_i = band.split(' ')
            band = split_i[1].rstrip('\n')
            uas_list.append(float(band))
    out_put = np.array(uas_list)
    return out_put

def export_result_to_netcdf(ds_array, filepath, output_folder):

    # open the location dadaset
    filename = str(filepath.name)
    ds = xr.open_dataset(str(filepath), engine='h5netcdf')
    loc = xr.open_dataset(str(filepath), group='location')

    # set the nodata value for glt and stack the x and y arrays together
    GLT_NODATA_VALUE = 0
    glt_array = np.nan_to_num(np.stack([loc['glt_x'].data, loc['glt_y'].data], axis=-1), nan=GLT_NODATA_VALUE).astype(
        int)

    # Build Output Dataset
    # the fill value is set to -9999
    fill_value = -9999
    # get an array with the same shape as the glt array and fill it with the fill value -9999
    out_ds = np.zeros((glt_array.shape[0], glt_array.shape[1]), dtype=np.float32) + fill_value

    # get an boolean array with the same shape as the glt array where the values are True if the glt array is not equal to the nodata value
    valid_glt = np.all(glt_array != GLT_NODATA_VALUE, axis=-1)
    # Adjust for One based Index
    # subtract 1 from the glt array where the valid_glt array is True
    glt_array[valid_glt] -= 1

    # Use indexing/broadcasting to populate array cells with 0 values
    out_ds[valid_glt] = ds_array[glt_array[valid_glt, 1], glt_array[valid_glt, 0]]

    # get the geotransform from the root dataset
    GT = ds.geotransform

    # Create Array for Lat and Lon and fill
    dim_x = loc.glt_x.shape[1]
    dim_y = loc.glt_x.shape[0]
    lon = np.zeros(dim_x)
    lat = np.zeros(dim_y)

    # fill the lat and lon arrays with the geotransform values
    for x in np.arange(dim_x):
        x_geo = (GT[0] + 0.5 * GT[1]) + x * GT[1]
        lon[x] = x_geo
    for y in np.arange(dim_y):
        y_geo = (GT[3] + 0.5 * GT[5]) + y * GT[5]
        lat[y] = y_geo

    ## ** upacks the existing dictionary from the wvl dataset.
    coords = {'lat': (['lat'], lat), 'lon': (['lon'], lon)}
    data_vars = {'methane_enhancement': (['lat', 'lon'], out_ds)}

    out_xr = xr.Dataset(data_vars=data_vars, coords=coords)
    out_xr.coords['lat'].attrs = loc['lat'].attrs
    out_xr.coords['lon'].attrs = loc['lon'].attrs
    out_xr.rio.write_crs(ds.spatial_ref, inplace=True)  # Add CRS in easily recognizable format
    output_path = output_folder + '/' + filename
    out_xr.to_netcdf(output_path)
    print(f"Exported to {output_path}")

def mf_process(filepath, uas_path, output_path, is_iterate=False):
    # open the unit absorption spectrum file and convert it a numpy array
    unit_absorption_spectrum = open_unit_absorption_spectrum(uas_path)

    # get the raster array from the radiance file
    radiance_data = np.array(read_nc_to_array(str(filepath)))
    radiance_window = radiance_data[radiance_data.shape[2]-len(unit_absorption_spectrum) + 1:-1]

    name = pl.Path(filepath).name.rstrip(".nc")
    # pre-define the list to store the band data and the count of the non-nan value


    # get the number of bands, rows and columns of the image data
    rows, cols, bands = radiance_window.shape

    # build the albedo, alpha to store the needing result
    u = np.zeros(cols, bands)
    target = np.zeros(cols, bands)
    albedo = np.zeros((rows, cols))
    alpha = np.zeros((rows, cols))
    # calculate the covariance matrix
    c = np.zeros((bands, bands))

    # alone the row axis to get the mean value of each band for each column
    for col_index in range(cols):
        count_not_nan = np.count_nonzero(~np.isnan(radiance_window[:, col_index, 0]))
        column_data = radiance_window[:, col_index, :]
        u[col_index,:] = np.nanmean(column_data, axia=0)
        # alone the row axis to calculate the covariance matrix
        for row in range(rows):
                if not np.isnan(radiance_window[row, col_index, 0]):
                    c += np.outer(radiance_window[row, col_index, :] - u[col_index,:],
                                  radiance_window[row, col_index, :] - u[col_index,:])

        # get the average of the sum of covariance matrix
        c = c / count_not_nan

        # get the inverse of the covariance matrix
        c_inverse = np.linalg.inv(c)

        # based on the background spectrum and the unit absorption spectrum, calculate the target spectrum
        target[col_index,:] = np.multiply(u[col_index,:], unit_absorption_spectrum)

        # iterate the whole region of the image data to calculate the albedo and the alpha for each pixel
        for row_index in range(rows):
                # account the nan value in the image data and make the result to be nan
                if not np.isnan(column_data[row_index, col_index, 0]):
                    # based on the formula to calculate the albedo of each pixel and store it into the albedo array
                    albedo[row_index, col_index] = (np.inner(column_data[row_index, col_index, :], u)
                                        /np.inner(u, u))
                    # based on the formula to calculate the methane enhancement of each pixel
                    # and store it in the alpha array
                    up = (column_data[row_index, col_index, :] - u) @ c_inverse @ target[col_index,:]
                    down = albedo[row_index, col_index] * (target[col_index,:] @ c_inverse @ target[col_index,:])
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
        for iter_num in range(20):
            iter_data = radiance_window.copy()
            for col_index in range(cols):
                for row_index in range(rows):
                    if not np.isnan(radiance_window[row_index, col_index, 0]):
                        iter_data[row_index, col_index, :] = (radiance_window[row_index, col_index, :] -
                                                              target[col_index,:] * alpha[row_index, col_index])
                        l1filter[row_index, col_index] = 1/(alpha[row_index, col_index] + epsilon)
            for col_index in range(cols):
                count_not_nan = np.count_nonzero(~np.isnan(iter_data[:, col_index, 0]))
                u[col_index,:] = np.nanmean(iter_data[:, col_index, :], axis=0)
                target[col_index,:] = np.multiply(u[col_index, :], unit_absorption_spectrum)
                # get the new covariance matrix
                c = np.zeros((bands, bands))
                for row in range(rows):
                    for col in range(cols):
                        if not np.isnan(radiance_window[row_index, col_index, 0]):
                            c += np.outer(radiance_window[row_index,col_index,:] -
                                          (u + albedo[row, col] * alpha[row, col] * target),
                                          radiance_window[row_index,col_index,:] -
                                          (u + albedo[row, col] * alpha[row, col] * target))
                c = c / count_not_nan
                # get the inverse of the covariance matrix
                c_inverse = np.linalg.inv(c)

                # calculate the new methane enhancement result
                for row in range(rows):
                    for col in range(cols):
                        if not np.isnan(radiance_window[row_index, col_index, 0]):
                            up = (radiance_window[row_index,col_index,:] - u) @ c_inverse @ target - l1filter[row, col]
                            down = albedo[row, col] * target[col_index,:] @ c_inverse @ target[col_index,:]
                            alpha[row, col] = max(up / down, 0)
                        else:
                            alpha[row, col] = np.nan

    # set the output tiff file path
    #  get the number of rows and columns of the alpha array
    result_rows, result_cols = alpha.shape
    output_folder = '/Users/nomoredrama/Local Documents/result'
    export_result_to_netcdf(alpha, filepath, output_folder)

# define the path of the unit absorption spectrum file and open it
uas_filepath = 'EMIT_unit_absorption_spectrum.txt'

# define the path of the radiance folder and get the radiance file list with an img suffix
radiance_folder = "F:\\EMIT_DATA\\envi"
radiance_path_list = pl.Path(radiance_folder).glob('*.nc')

# get the output file path and get the existing output file list to avoid the repeat process
root = pl.Path("F:\\EMIT_DATA\\result")
output = root.glob('*.nc')
outputfile = []
for i in output:
    outputfile.append(str(i.name))

# define the main function to process the radiance file by using the matching filter algorithm
# the input includes the radiance file path, the unit absorption spectrum, the output path and the is_iterate flag

# iterate the radiance file list to process the radiance file by using the matching filter algorithm
for radiance_path in radiance_path_list:
    current_filename = str(radiance_path.name)
    if current_filename in outputfile:
        continue
    else:
        print(f"{current_filename} is now being processed")
        try :
            mf_process(radiance_path, uas_path=uas_filepath, output_path=root, is_iterate=False)
            print(f"{current_filename} has been processed")
        except Exception as e:
            print(f"{current_filename} has an error")
            print(e)
            pass

