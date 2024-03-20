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


# define the main function to process the radiance file by using the matching filter algorithm
def mf_process(filepath, uas_path, output_path, is_iterate=False, is_albedo=False, is_filter=False):
    # open the unit absorption spectrum file and convert it a numpy array
    unit_absorption_spectrum = open_unit_absorption_spectrum(uas_path)

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
    export_result_to_netcdf(alpha, filepath, output_folder)

        # maybe directly convert into tiff file

# define the function to export the methane enhancement result to a nc file
def export_result_to_netcdf(ds_array, filepath, output_folder):

    # open the location dadaset
    filename = str(filepath.name)
    ds = xr.open_dataset(str(filepath), engine='h5netcdf')
    loc = xr.open_dataset(str(filepath), group='location')

    # set the nodata value for glt and stack the x and y arrays together
    glt_nodata_value = 0
    glt_array = np.nan_to_num(np.stack([loc['glt_x'].data, loc['glt_y'].data], axis=-1), nan=glt_nodata_value).astype(
        int)

    # Build Output Dataset
    # the fill value is set to -9999
    fill_value = -9999
    # get an array with the same shape as the glt array and fill it with the fill value -9999
    out_ds = np.zeros((glt_array.shape[0], glt_array.shape[1]), dtype=np.float32) + fill_value

    # get an boolean array with the same shape as the glt array where the values are True if the glt array is not equal to the nodata value
    valid_glt = np.all(glt_array != glt_nodata_value, axis=-1)
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

    out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)
    out_xr.coords['lat'].attrs = loc['lat'].attrs
    out_xr.coords['lon'].attrs = loc['lon'].attrs
    out_xr.rio.write_crs(ds.spatial_ref, inplace=True)  # Add CRS in easily recognizable format
    output_path = output_folder + '/' + filename
    out_xr.to_netcdf(output_path)
    print(f"Exported to {output_path}")

# define the path of the unit absorption spectrum file and open it
uas_filepath = 'EMIT_unit_absorption_spectrum.txt'

# define the path of the radiance folder and get the radiance file list with an img suffix
radiance_folder = "I:\\EMIT\\rad"
radiance_path_list = pl.Path(radiance_folder).glob('*.nc')

# get the output file path and get the existing output file list to avoid the repeat process
root = pl.Path("I:\\EMIT\\methane_result\\direct")
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
