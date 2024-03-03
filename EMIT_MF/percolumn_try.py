import xarray as xr
import numpy as np
filepath = "/Users/nomoredrama/Downloads/EMIT_L1B_RAD_001_20220814T051412_2222604_005.nc"
radiance = xr.open_dataset(filepath)
loc = xr.open_dataset(filepath, group='location')
lon = loc['lon']
lat = loc['lat']
radiance = radiance['radiance']
radiance = np.array(radiance)
new_radiance = radiance.transpose(2, 0, 1)

background_spectrum = np.nanmean(new_radiance, axis=(1, 2))


