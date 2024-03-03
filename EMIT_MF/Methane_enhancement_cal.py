"""
this code is used to process the radiance file by using the matching filter algorithm
and the goal is to get the methane enhancement image
"""

# the necessary lib to be imported
import numpy as np
from osgeo import gdal
import pathlib as pl


# a function to get the raster array and return the dataset
def get_raster_array(filepath):
    # open the dataset from the filepath
    dataset = gdal.Open(filepath, gdal.GA_ReadOnly)
    return dataset


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


def mf_process(filepath, uas_path, output_path, is_iterate=False):
    # get the file name to make the output path string
    name = filepath.name.rstrip("radiance.img")

    # open the unit absorption spectrum file and convert it a numpy array
    unit_absorption_spectrum = open_unit_absorption_spectrum(uas_path)

    # get the raster array from the radiance file
    dataset = get_raster_array(str(filepath))

    # get the basic information of the raster file such as the geo transform, the projection and the number of bands
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    num_bands = dataset.RasterCount

    # pre-define the list to store the band data and the count of the non-nan value
    band_data_list = []
    count_not_nan = 0

    # iterate the bands to get the band data and the count of the non-nan value
    for band_index in range(0, len(unit_absorption_spectrum)):
        # based on the band index, get the dataset of the image file
        band = dataset.GetRasterBand(num_bands - len(unit_absorption_spectrum) + band_index + 1)
        # convert the dataset into the numpy array
        band_data = band.ReadAsArray()
        # append the band data into the band data list
        band_data_list.append(band_data)
        # sum up all the non-nan value in the band data
        if band_index == 0:
            count_not_nan = np.count_nonzero(~np.isnan(band_data))

    # convert the band data list into a 3-D numpy array
    image_data = np.array(band_data_list)
    # get the number of bands, rows and columns of the image data
    bands, rows, cols = image_data.shape

    # alone the row and column axis to get the mean value of each band and
    # the shape of the result should be the number of bands
    u = np.nanmean(image_data, axis=(1, 2))

    # build the albedo, alpha to store the needing result
    albedo = np.zeros((rows, cols))
    alpha = np.zeros((rows, cols))

    # calculate the covariance matrix
    c = np.zeros((bands, bands))
    for row in range(rows):
        for col in range(cols):
            if not np.isnan(image_data[0, row, col]):
                c += np.outer(image_data[:, row, col] - u, image_data[:, row, col] - u)
    # get the average of the sum of covariance matrix
    c = c / count_not_nan
    # get the inverse of the covariance matrix
    c_inverse = np.linalg.inv(c)

    # based on the background spectrum and the unit absorption spectrum, calculate the target spectrum
    target = np.multiply(u, unit_absorption_spectrum)

    # iterate the whole region of the image data to calculate the albedo and the alpha for each pixel
    for row in range(rows):
        for col in range(cols):
            # account the nan value in the image data and make the result to be nan
            if not np.isnan(image_data[0, row, col]):
                # based on the formula to calculate the albedo of each pixel and store it into the albedo array
                albedo[row, col] = (np.inner(image_data[:, row, col], u)
                                    / np.inner(u, u))
                # based on the formula to calculate the methane enhancement of each pixel
                # and store it in the alpha array
                up = (image_data[:, row, col] - u) @ c_inverse @ target
                down = albedo[row, col] * (target @ c_inverse @ target)
                alpha[row, col] = up / down
            else:
                alpha[row, col] = np.nan

    if is_iterate:
        # build the l1_filter to store the result
        l1filter = np.zeros((rows, cols))
        # define a tiny value to avoid the zero division
        epsilon = np.finfo(np.float32).tiny
        # based on the former alpha result, update the background spectrum and the target spectrum
        # and then update the covariance matrix
        # and get the new methane enhancement result
        for iter_num in range(20):
            iter_data = image_data.copy()
            for row in range(rows):
                for col in range(cols):
                    if not np.isnan(image_data[0, row, col]):
                        iter_data[:, row, col] = image_data[:, row, col] - target * alpha[row, col]
                        l1filter[row, col] = 1/(alpha[row, col] + epsilon)
            # get the new background spectrum and the target spectrum
            u = np.nanmean(iter_data, axis=(1, 2))
            target = np.multiply(u, unit_absorption_spectrum)
            # get the new covariance matrix
            c = np.zeros((bands, bands))
            for row in range(rows):
                for col in range(cols):
                    if not np.isnan(image_data[0, row, col]):
                        c += np.outer(image_data[:, row, col] - (u + albedo[row, col] * alpha[row, col] * target),
                                      image_data[:, row, col] - (u + albedo[row, col] * alpha[row, col] * target))
            c = c / count_not_nan
            # get the inverse of the covariance matrix
            c_inverse = np.linalg.inv(c)

            # calculate the new methane enhancement result
            for row in range(rows):
                for col in range(cols):
                    if not np.isnan(image_data[0, row, col]):
                        up = (image_data[:, row, col] - u) @ c_inverse @ target - l1filter[row, col]
                        down = albedo[row, col] * target @ c_inverse @ target
                        alpha[row, col] = max(up / down, 0)
                    else:
                        alpha[row, col] = np.nan

    # set the output tiff file path
    output_tiff_file = str(output_path / (name + 'enhancement.tif'))

    #  get the number of rows and columns of the alpha array
    result_rows, result_cols = alpha.shape

    # create a new raster file
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_tiff_file, result_cols, result_rows, 1, gdal.GDT_Float32)

    # write the data to a single band raster
    band = dataset.GetRasterBand(1)
    band.WriteArray(alpha)

    # set the geo_transform and projection of the dataset
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)


# define the path of the unit absorption spectrum file and open it
uas_filepath = 'EMIT_unit_absorption_spectrum.txt'

# define the path of the radiance folder and get the radiance file list with an img suffix
radiance_folder = "F:\\EMIT_DATA\\envi"
radiance_path_list = pl.Path(radiance_folder).glob('*.img')

# get the output file path and get the existing output file list to avoid the repeat process
root = pl.Path("F:\\EMIT_DATA\\result")
output = root.glob('*.tif')
outputfile = []
for i in output:
    outputfile.append(str(i.name))

# define the main function to process the radiance file by using the matching filter algorithm
# the input includes the radiance file path, the unit absorption spectrum, the output path and the is_iterate flag

# iterate the radiance file list to process the radiance file by using the matching filter algorithm
for radiance_path in radiance_path_list:
    current_filename = str(radiance_path.name.rstrip("radiance.img") + "enhancement.tif")
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

