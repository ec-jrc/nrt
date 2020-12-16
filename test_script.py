import pkg_resources
import datetime
import tempfile
import os

import xarray as xr
import numpy as np
from numpy.testing import assert_array_equal
from netCDF4 import Dataset

from nrt.iqr import Iqr

TMP = tempfile.gettempdir()
filename = os.path.join(TMP, 'romania_iqr_monitor.nc')

ds_filename = pkg_resources.resource_filename('nrt', 'data/sentinel2_cube_subset_romania_20m.nc')
ds = xr.open_dataset(ds_filename)
print(ds) # for some reason data variables are already in float, no so sure why
ds['ndvi'] = (ds.B8A - ds.B4) / (ds.B8A + ds.B4)
ds = ds.where(ds.SCL.isin([4,5,7]))
da_history = ds.ndvi.sel(time=slice('2015-01-01', '2017-12-31'))
da_monitor = ds.ndvi.sel(time=slice('2018-01-01', '2020-12-31'))

iqr_romania = Iqr(sensitivity=0.7)
iqr_romania.fit(da_history)

iqr_romania.to_netcdf(filename)

# One week later
iqr_romania_new = Iqr.from_netcdf(filename)


count = 0
for array_, date_ in zip(da_monitor.values, da_monitor.time.values):
    count += 1
    iqr_romania_new.monitor(array_, date_)
    if count % 10 == 0:
        print(iqr_romania_new._report())

# TODO:
    # Need debugging, script does not work because of Nan
    # self.beta only has NA, so start from there
