import pkg_resources
import datetime

import pytest
import xarray as xr
import rasterio
import numpy as np
import pandas as pd

@pytest.fixture
def history_dataarray():
    """History dataarray over romania

    1 squared km over a forest in Romania.
    NDVI with cloud already filtered (appear as np.nan) in the arrays
    3.5 years of data from 2015 to end of 2018
    """
    filename = pkg_resources.resource_filename('nrt',
                                               'data/sentinel2_cube_subset_romania_20m.nc')
    ds = xr.open_dataset(filename)
    ds['ndvi'] = (ds.B8A - ds.B4) / (ds.B8A + ds.B4)
    ds = ds.where(ds.SCL.isin([4,5,7]))
    history = ds.sel(time=slice(datetime.datetime(2015, 1, 1),
                                     datetime.datetime(2018, 12, 31)))
    return history


@pytest.fixture
def ndvi_history(history_dataarray):
    """A NDVI dataarray of Romania
    """
    return history_dataarray.ndvi


@pytest.fixture
def green_swir_history(history_dataarray):
    """A NDVI dataarray of Romania
    """
    return history_dataarray.B3, history_dataarray.B11


@pytest.fixture
def ndvi_monitoring_numpy():
    filename = pkg_resources.resource_filename('nrt',
                                               'data/sentinel2_cube_subset_romania_20m.nc')
    ds = xr.open_dataset(filename)
    ds['ndvi'] = (ds.B8A - ds.B4) / (ds.B8A + ds.B4)
    ds = ds.where(ds.SCL.isin([4,5,7]))
    ndvi_monitoring = ds.ndvi.sel(time=slice(datetime.datetime(2019, 1, 1),
                                             datetime.datetime(2021, 1, 15)))
    values = ndvi_monitoring.values
    dates = ndvi_monitoring.time.values.astype('datetime64[s]').tolist()
    return values, dates


@pytest.fixture
def forest_mask():
    """Forest density over romania
    """
    filename = pkg_resources.resource_filename('nrt',
                                               'data/tree_cover_density_2018_romania.tif')
    with rasterio.open(filename) as src:
        arr = src.read(1)
    return (arr > 30).astype(np.int8)
