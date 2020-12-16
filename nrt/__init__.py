import abc

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import rasterio
from rasterio.crs import CRS
from affine import Affine

from nrt.utils import build_regressors

__version__ = "0.0.1"


class BaseNrt(metaclass=abc.ABCMeta):
    """Abstract class for Near Real Time change detection

    Every new change monitoring approach should inherit from this abstract
    class and must implement the abstract methods ``fit()``, ``monitor()``
    and ``report()``. It contains generic method to fit time-series
    harmonic-trend regression models, backward test for stability, dump the
    instance to a netcdf file and reload a dump.

    Attributes:
        mask (numpy.ndarray): A 2D numpy array containing pixels that should
            be monitored (1) and not (0). The mask may be updated following
            historing period stability check, and after a call to monitor
            following a confirmed break. Values are as follow.
            ``{0: 'Not monitored', 1: 'monitored', 2: 'Unstable history',
            3: 'Confirmed break - no longer monitored'}``
        trend (bool): Indicate whether stable period fit is performed with
            trend or not
        harmonic_order (int): The harmonic order of the time-series regression
        x (numpy.ndarray): array of x coordinates
        y (numpy.ndarray): array of y coordinates

    Args:
        mask (numpy.ndarray): A 2D numpy array containing pixels that should be
            monitored marked as ``1`` and pixels that should be excluded (marked
            as ``0``). Typically a stable forest mask when doing forest disturbance
            monitoring. If no mask is supplied all pixels are considered and
            a mask is created following the ``fit()`` call
        trend (bool): Indicate whether stable period fit is performed with
            trend or not
        harmonic_order (int): The harmonic order of the time-series regression
        x_coords (numpy.ndarray): x coordinates
        y_coords (numpy.ndarray): y coordinates
    """
    def __init__(self, mask=None, trend=True, harmonic_order=3, beta=None,
                 x_coords=None, y_coords=None):
        self.mask = mask
        self.trend = trend
        self.harmonic_order = harmonic_order
        self.x = x_coords
        self.y = y_coords
        self.beta = beta

    def _fit(self, X, dataarray, reg='OLS', check_stability=None, **kwargs):
        """Fit a regression model on an xarray.DataArray

        #TODO: Not sure whether recresid is implied by ROC or not.

        Args:
            X (numpy.ndarray): The design matrix used for the regression
            dataarray (xarray.DataArray): A 3 dimension (time, y, x) DataArray containing
                the dependant variable
            reg (str): The regression type. Possible values include ``'OLS'``,
                ``'IRLS'``, ``'LASSO'``. May be ignored depending on the value
                passed to ``check_stability``
            check_stability (str): Which test should be used in stability checking.
                If ``None`` no stability check is performed. Other potential values
                include ``'ROC'``.
            **kwargs: Other parameters specific to each regression type

        Returns:
            beta (numpy.ndarray): The array of regression estimators
            residuals (numpy.ndarray): The array of residuals
        """
        y = dataarray.values
        shape = y.shape
        # TODO: Implement mask subsetting
        shape_flat = (shape[0], shape[1]*shape[2])
        beta_shape = (X.shape[1], shape[1], shape[2])
        y_flat = y.reshape(shape_flat)
        if reg == 'OLS' and not check_stability:
            beta, _, _, _ = np.linalg.lstsq(X, y_flat)
            residuals = np.dot(X, beta) - y_flat
        elif reg == 'LASSO' and not check_stability:
            raise NotImplementedError('Regression type not yet implemented')
        else:
            raise ValueError('Unknown regression type')
        residuals = residuals.reshape(shape)
        beta = beta.reshape(beta_shape)
        return beta, residuals

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def monitor(self):
        pass

    @abc.abstractmethod
    def _report(self):
        """Abstract method

        Must generate a 2D numpy array with unit8 datatype
        """
        pass

    def report(self, filename, driver='GTiff', crs=CRS.from_epsg(3035)):
        """Write the result of reporting to a raster geospatial file

        TODO: Make writing a window to larger file possible, but check for potential
            thread safety issues
        """
        r = self._report()
        meta = {'driver': driver,
                'crs': crs,
                'count': 1,
                'dtype': 'uint8',
                'transform': self.affine,
                'height': r.shape[0],
                'width': r.shape[1]}
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(r, 1)

    @property
    def affine(self):
        # TODO compute affine from x and y coord arrays
        return Affine.identity()

    def predict(self, date):
        """Predict the expected values for a given date

        Args:
            date (datetime.datetime): The date to predict

        Returns:
            numpy.ndarray: A 2D array of predicted values
        """
        shape_beta = self.beta.shape
        shape_y = (shape_beta[1], shape_beta[2])
        shape_beta_flat = (shape_beta[0], shape_beta[1] * shape_beta[2])
        X = self._regressors(date)
        y_pred = np.dot(X, self.beta.reshape(shape_beta_flat))
        return y_pred.reshape(shape_y)

    @classmethod
    def from_netcdf(cls, filename, **kwargs):
        with Dataset(filename) as src:
            # Get dict of variables
            d = dict()
            for k in src.variables.keys():
                print(k)
                try:
                    v = src.variables[k].value
                except Exception as e:
                    v = src.variables[k][:]
                if k == 'x':
                    k = 'x_coords'
                if k == 'y':
                    k = 'y_coords'
                if k == 'r':
                    continue
                d.update({k:v})
        return cls(**d)

    def to_netcdf(self, filename):
        # List all attributes remove
        attr = vars(self)
        with Dataset(filename, 'w') as dst:
            # define variable
            x_dim = dst.createDimension('x', len(self.x))
            y_dim = dst.createDimension('y', len(self.y))
            r_dim = dst.createDimension('r', self.beta.shape[0])
            # Create coordinate variables
            x_var = dst.createVariable('x', np.float32, ('x',))
            y_var = dst.createVariable('y', np.float32, ('y',))
            r_var = dst.createVariable('r', np.uint8, ('r', ))
            # fill values of coordinate variables
            x_var[:] = self.x
            y_var[:] = self.y
            r_var[:] = np.arange(start=0, stop=self.beta.shape[0],
                                 dtype=np.uint8)
            # Add beta variable
            beta_var = dst.createVariable('beta', np.float32, ('r', 'y', 'x'),
                                          zlib=True)
            beta_var[:] = self.beta
            # Create other variables
            for k,v in attr.items():
                if k not in ['x', 'y', 'beta']:
                    if isinstance(v, np.ndarray):
                        new_var = dst.createVariable(k, v.dtype, ('y', 'x'))
                        new_var[:] = v
                    elif isinstance(v, str):
                        new_var = dst.createVariable(k, 'c')
                        new_var.value = v
                    elif isinstance(v, float):
                        new_var = dst.createVariable(k, 'f4')
                        new_var.value = v
                    elif isinstance(v, bool):
                        new_var = dst.createVariable(k, 'i1')
                        new_var.value = int(v)
                    elif isinstance(v, int):
                        new_var = dst.createVariable(k, 'i4')
                        new_var.value = v


    def set_xy(self, dataarray):
        self.x = dataarray.x.values
        self.y = dataarray.y.values

    @staticmethod
    def build_design_matrix(dataarray, trend=True, harmonic_order=3):
        """Build a design matrix for temporal regression from xarray DataArray

        Args:
            trend (bool): Whether to include a temporal trend or not
            harmonic_order (int): The harmonic order to use (``1`` corresponds
                to annual cycles, ``2`` to annual and biannual cycles, etc)

        Returns:
            numpy.ndarray: A design matrix to be passed to be passed to e.g. the
                ``_fit()`` method
        """
        dates = pd.DatetimeIndex(dataarray.time.values)
        X = build_regressors(dates=dates, trend=trend, harmonic_order=harmonic_order)
        return X

    def _regressors(self, date):
        """Get the matrix of regressors for a single date

        Args:
            date (datetime.datetime): The date for which to generate a matrix of regressors

        Returns:
            numpy.ndarray: A matrix of regressors
        """
        date_pd = pd.DatetimeIndex([date])
        X = build_regressors(dates=date_pd, trend=self.trend,
                             harmonic_order=self.harmonic_order)
        return X


