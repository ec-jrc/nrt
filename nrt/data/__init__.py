import os
import pickle

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

def mre_crit_table():
    """Contains a dictionary equivalent to strucchange's ``mreCritValTable``
    Contains a tuple, where the first variable is a list of the available pre-
    computed significance (1-alpha) values.

    The second variable is a nested dictionary, where the first dict are the
    available relative window sizes (0.25, 0.5, 1), the second dict are the
    available periods (2, 4, 6, 8, 10) and the third dict is the functional type
    ("max", "range").

    For example:
        >>> sig_level, crit_dict = data.mre_crit_table()
        >>> win_size = 0.5
        >>> period = 10
        >>> functional = "max"
        >>> alpha=0.025
        >>> crit_values = crit_dict.get(str(win_size))\
        ...                        .get(str(period))\
        ...                        .get(functional)
        >>> crit_level = np.interp(1-alpha, sig_level, crit_values)
    """
    with open(os.path.join(data_dir, "mreCritValTable.bin"), "rb") as crit:
        crit_table = pickle.load(crit)
    return crit_table
