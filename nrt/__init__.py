import abc
import warnings

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import rasterio
from rasterio.crs import CRS
from affine import Affine

from nrt.utils import build_regressors
from nrt.fit_methods import ols, rirls, ccdc_stable_fit, roc_stable_fit
from nrt.outliers import ccdc_rirls, shewhart
from nrt.cusum import _cusum_rec_test_crit

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

    def _fit(self, X, dataarray,
             method='OLS',
             screen_outliers=None, **kwargs):
        """Fit a regression model on an xarray.DataArray

        #TODO: Not sure whether recresid is implied by ROC or not.

        Args:
            X (numpy.ndarray): The design matrix used for the regression
            dataarray (xarray.DataArray): A 3 dimension (time, y, x) DataArray
                containing the dependant variable
            method (str): The fitting method. Possible values include ``'OLS'``,
                ``'IRLS'``, ``'LASSO'`` and ``'CCDC-stable'``.
            screen_outliers (str): The screening method. Possible values include
                ``'Shewhart'`` and ``'CCDC_RIRLS'``.
            **kwargs: Other parameters specific to each fit method

        Returns:
            beta (numpy.ndarray): The array of regression estimators
            residuals (numpy.ndarray): The array of residuals

        Raises:
            NotImplementedError: If method is not yet implemented
            ValueError: Unknown value for `method`
        """
        # lower level functions using numba may require that X and y have the same
        # datatype (e.g. float32, float32 signature)
        y = dataarray.values.astype(np.float32)
        X = X.astype(np.float32)
        # If no mask has been set at class instantiation, assume everything is forest
        if self.mask is None:
            self.mask = np.ones_like(y[0,:,:], dtype=np.uint8)
        mask_bool = self.mask == 1
        shape = y.shape
        beta_shape = (X.shape[1], shape[1], shape[2])
        # Create empty arrays with output shapes to store reg coefficients and residuals
        beta = np.zeros(beta_shape, dtype=np.float32)
        residuals = np.zeros_like(y, dtype=np.float32)
        y_flat = y[:, mask_bool]

        # 1. Optionally screen outliers
        #   This just updats y_flat
        if screen_outliers == 'Shewhart':
            y_flat = shewhart(X, y_flat, **kwargs)
        elif screen_outliers == 'CCDC_RIRLS':
            try:
                green_flat = kwargs.pop('green').values\
                    .astype(np.float32)[:, mask_bool]
                swir_flat = kwargs.pop('swir').values\
                    .astype(np.float32)[:, mask_bool]
            except KeyError as e:
                raise ValueError('Parameters `green` and `swir` need to be '
                           'passed for CCDC_RIRLS.')
            y_flat = ccdc_rirls(X, y_flat,
                                green=green_flat, swir=swir_flat, **kwargs)
        elif screen_outliers:
            raise ValueError('Unknown screen_outliers')

        # 2. Fit using specified method
        if method == 'ROC':
            try:
                alpha = kwargs.pop('alpha')
            except KeyError as e:
                warnings.warn('Parameter `alpha` needs to be '
                              'passed for ROC fit. Using alpha of 0.05.')
                alpha = 0.05
            # Convert datetime64 to days, so numba is happy
            dates = dataarray.time.values.astype('datetime64[D]').astype('int')
            # crit already calculated here, to allow numba in roc_stable_fit
            crit = _cusum_rec_test_crit(alpha)
            beta_flat, residuals_flat, is_stable = \
                roc_stable_fit(X, y_flat, dates,
                               alpha=alpha, crit=crit)
            is_stable_2d = is_stable.reshape(y.shape[1], y.shape[2])
            self.mask[~is_stable_2d] = 2
        elif method == 'CCDC-stable':
            if not self.trend:
                raise ValueError('Method "CCDC" requires "trend" to be true.')
            dates = dataarray.time.values
            beta_flat, residuals_flat, is_stable = \
                ccdc_stable_fit(X, y_flat, dates, **kwargs)
            is_stable_2d = is_stable.reshape(y.shape[1], y.shape[2])
            self.mask[~is_stable_2d] = 2
        elif method == 'OLS':
            beta_flat, residuals_flat = ols(X, y_flat)
        elif method == 'LASSO':
            raise NotImplementedError('Method not yet implemented')
        elif method == 'RIRLS':
            beta_flat, residuals_flat = rirls(X, y_flat, **kwargs)
        else:
            raise ValueError('Unknown method')

        beta[:, mask_bool] = beta_flat
        residuals[:, mask_bool] = residuals_flat
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
                'transform': self.transform,
                'height': r.shape[0],
                'width': r.shape[1]}
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(r, 1)

    @property
    def transform(self):
        if self.x is None or self.y is None:
            warnings.warn('x and y coordinate arrays not set, returning identity transform')
            aff = Affine.identity()
        else:
            y_res = abs(self.y[0] - self.y[1])
            x_res = abs(self.x[0] - self.x[1])
            y_0 = np.max(self.y) + y_res / 2
            x_0 = np.min(self.x) - x_res / 2
            aff = Affine(x_res, 0, x_0,
                         0, -y_res, y_0)
        return aff

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
                nc_var = src.variables[k]
                # bool are stored as int in netcdf and need to be coerced back to bool
                is_bool = 'dtype' in nc_var.ncattrs() and nc_var.getncattr('dtype') == 'bool'
                try:
                    v = nc_var.value
                    if is_bool:
                        v = bool(v)
                except Exception as e:
                    v = nc_var[:]
                    if is_bool:
                        v = v.astype(np.bool)
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
                        # bool array are stored as int8
                        dtype = np.uint8 if v.dtype == bool else v.dtype
                        new_var = dst.createVariable(k, dtype, ('y', 'x'))
                        new_var[:] = v
                        if v.dtype == bool:
                            new_var.setncattr('dtype', 'bool')
                    elif isinstance(v, str):
                        new_var = dst.createVariable(k, 'c')
                        new_var.value = v
                    elif isinstance(v, float):
                        new_var = dst.createVariable(k, 'f4')
                        new_var.value = v
                    elif isinstance(v, bool):
                        new_var = dst.createVariable(k, 'i1')
                        new_var.value = int(v)
                        new_var.setncattr('dtype', 'bool')
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


