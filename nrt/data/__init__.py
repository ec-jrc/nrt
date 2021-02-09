import os

import xarray as xr
import rasterio


data_dir = os.path.abspath(os.path.dirname(__file__))


def _load(f):
    """Load a ncdf file located in the data directory as a xarray Dataset

    Args:
        f (str): File basename

    Return:
        xarray.Dataset: The Dataset
    """
    xr_dataset = xr.open_dataset(os.path.join(data_dir, f))
    return xr_dataset


def romania_10m():
    """Sentinel 2 datacube of a small forested area in Romania at 10 m resolution

    Examples:
        >>> from nrt import data

        >>> s2_cube = data.romania_10m()
        >>> # Compute NDVI
        >>> s2_cube['ndvi'] = (s2_cube.B8 - s2_cube.B4) / (s2_cube.B8 + s2_cube.B4)
        >>> # Filter clouds
        >>> s2_cube = s2_cube.where(s2_cube.SCL.isin([4,5,7]))
    """
    return _load('sentinel2_cube_subset_romania_10m.nc')


def romania_20m():
    """Sentinel 2 datacube of a small forested area in Romania at 20 m resolution

    Examples:
        >>> from nrt import data

        >>> s2_cube = data.romania_20m()
        >>> # Compute NDVI
        >>> s2_cube['ndvi'] = (s2_cube.B8A - s2_cube.B4) / (s2_cube.B8A + s2_cube.B4)
        >>> # Filter clouds
        >>> s2_cube = s2_cube.where(s2_cube.SCL.isin([4,5,7]))
    """
    return _load('sentinel2_cube_subset_romania_20m.nc')


def romania_forest_cover_percentage():
    """Subset of Copernicus HR layer tree cover percentage - 20 m - Romania
    """
    file_basename = 'tree_cover_density_2018_romania.tif'
    filename = os.path.join(data_dir, file_basename)
    with rasterio.open(filename) as src:
        arr = src.read(1)
    return arr

