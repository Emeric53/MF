import numpy as np
import xarray as xr
import sys
import pathlib

#set the file path of the nc file
filepath = "F:\\EMIT_DATA\\nc\\EMIT_L1B_RAD_001_20220810T064957_2222205_033.nc"

# open the root dataset and the location dadaset
ds = xr.open_dataset(filepath, engine='h5netcdf')
loc = xr.open_dataset(filepath, group='location')
print(loc['glt_x'].data.shape)

# set the nodata value for glt and stack the x and y arrays together
GLT_NODATA_VALUE = 0
glt_array = np.nan_to_num(np.stack([loc['glt_x'].data, loc['glt_y'].data], axis=-1), nan=GLT_NODATA_VALUE).astype(int)
print(glt_array.shape)

# get the radiance array from the root dataset
ds_array = ds['radiance'].data
ds_array = ds_array[:,:,0]
print(ds_array.shape)

# Build Output Dataset
# the fill value is set to -9999
fill_value = -9999
# get an array with the same shape as the glt array and fill it with the fill value -9999
out_ds = np.zeros((glt_array.shape[0], glt_array.shape[1]), dtype=np.float32) + fill_value
print(out_ds.shape)

# get an boolean array with the same shape as the glt array where the values are True if the glt array is not equal to the nodata value
valid_glt = np.all(glt_array != GLT_NODATA_VALUE, axis=-1)
print(valid_glt.shape)
# Adjust for One based Index
print(glt_array)
# subtract 1 from the glt array where the valid_glt array is True
glt_array[valid_glt] -= 1
print(glt_array)

# Use indexing/broadcasting to populate array cells with 0 values
out_ds[valid_glt] = ds_array[glt_array[valid_glt, 1], glt_array[valid_glt, 0]]
print(out_ds.shape)

# get the geotransform from the root dataset
GT = ds.geotransform

# Create Array for Lat and Lon and fill
dim_x = loc.glt_x.shape[1]
dim_y = loc.glt_x.shape[0]
lon = np.zeros(dim_x)
lat = np.zeros(dim_y)

# fill the lat and lon arrays with the geotransform values
for x in np.arange(dim_x):
    x_geo = (GT[0]+0.5*GT[1]) + x * GT[1]
    lon[x] = x_geo
for y in np.arange(dim_y):
    y_geo = (GT[3]+0.5*GT[5]) + y * GT[5]
    lat[y] = y_geo

print(lat, lon)

## ** upacks the existing dictionary from the wvl dataset.
coords = {'lat':(['lat'],lat), 'lon':(['lon'],lon)}
data_vars = {'radiance': (['lat','lon'], out_ds)}

out_xr = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds.attrs)
out_xr['radiance'].attrs = ds[('radiance')].attrs
out_xr.coords['lat'].attrs = loc['lat'].attrs
out_xr.coords['lon'].attrs = loc['lon'].attrs
out_xr.rio.write_crs(ds.spatial_ref, inplace=True) # Add CRS in easily recognizable format
out_xr.to_netcdf("C:\\Users\\RS\\Desktop\\export1.nc")
