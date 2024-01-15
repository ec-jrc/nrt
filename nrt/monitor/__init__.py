# Copyright (C) 2022 European Union (Joint Research Centre)
#
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#   https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.

import abc
import warnings
import datetime

import numpy as np
import pandas as pd
import numba
from netCDF4 import Dataset
import rasterio
from rasterio.crs import CRS
from affine import Affine

from nrt.utils import build_regressors
from nrt.fit_methods import ols, rirls, ccdc_stable_fit, roc_stable_fit
from nrt.outliers import ccdc_rirls, shewhart
from nrt.utils_efp import _cusum_rec_test_crit


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
            ``{0: 'Not monitored',
               1: 'monitored',
               2: 'Unstable history',
               3: 'Confirmed break - no longer monitored',
               4: 'Not enough observations - not monitored'}``
        trend (bool): Indicate whether stable period fit is performed with
            trend or not
        harmonic_order (int): The harmonic order of the time-series regression
        x (numpy.ndarray): array of x coordinates
        y (numpy.ndarray): array of y coordinates
        process (numpy.ndarray): 2D numpy array containing the
            process value for every pixel
        boundary (Union[numpy.ndarray, int, float]): Process boundary for all
            pixels or every pixel individually
        detection_date (numpy.ndarray): 2D array signalling detection date of
            disturbances in days since 1970-01-01
        fit_start (numpy.ndarray): 2D integer array reporting start of history
            period in days since UNIX Epoch. Start of history period only varies
            when using stable fitting algorithms

    Args:
        mask (numpy.ndarray): A 2D numpy array containing pixels that should be
            monitored marked as ``1`` and pixels that should be excluded (marked
            as ``0``). Typically a stable forest mask when doing forest disturbance
            monitoring. If no mask is supplied all pixels are considered and
            a mask is created following the ``fit()`` call
        trend (bool): Indicate whether stable period fit is performed with
            trend or not
        harmonic_order (int): The harmonic order of the time-series regression
        save_fit_start (bool): If start of the fit should be reported in the
            model. Only applicable to stable fits (e.g. 'ROC', 'CCDC-stable').
            If true, the data will be saved in the attribute `fit_start`
        x_coords (numpy.ndarray): x coordinates
        y_coords (numpy.ndarray): y coordinates
        process (numpy.ndarray): 2D numpy array containing the
            process value for every pixel
        boundary (Union[numpy.ndarray, int, float]): Process boundary for all
            pixels or every pixel individually
        detection_date (numpy.ndarray): 2D array signalling detection date of
            disturbances in days since 1970-01-01
        fit_start (numpy.ndarray): 2D integer array reporting start of history
            period in days since UNIX Epoch. Start of history period only varies
            when using stable fitting algorithms
    """
    def __init__(self, mask=None, trend=True, harmonic_order=3,
                 save_fit_start=False, beta=None, x_coords=None, y_coords=None,
                 process=None, boundary=None, detection_date=None,
                 fit_start=None, **kwargs):
        self.mask = np.copy(mask) if isinstance(mask, np.ndarray) else mask
        self.trend = trend
        self.harmonic_order = harmonic_order
        self.x = x_coords
        self.y = y_coords
        self.beta = beta
        self.process = process
        self.boundary = boundary
        self.detection_date = detection_date
        if save_fit_start:
            self.fit_start = fit_start

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        try:
            if vars(self).keys() != vars(other).keys():
                return False
            for key, value in vars(self).items():
                if isinstance(value, np.ndarray):
                    is_equal = np.array_equal(value, getattr(other, key),
                                              equal_nan=True)
                else:
                    is_equal = value == getattr(other, key)
                if not is_equal:
                    return False
            return True
        except AttributeError:
            return False

    def _fit(self, X, dataarray,
             method='OLS',
             screen_outliers=None,
             n_threads=1, **kwargs):
        """Fit a regression model on an xarray.DataArray
        Args:
            X (numpy.ndarray): The design matrix used for the regression
            dataarray (xarray.DataArray): A 3 dimension (time, y, x) DataArray
                containing the dependant variable
            method (str): The fitting method. Possible values include ``'OLS'``,
                ``'RIRLS'``, ``'LASSO'``, ``'ROC'`` and ``'CCDC-stable'``.
            screen_outliers (str): The screening method. Possible values include
                ``'Shewhart'`` and ``'CCDC_RIRLS'``.
            n_threads (int): Number of threads used for parallel fitting. Note that
                parallel fitting is not supported for ``ROC``; and that argument
                has therefore no impact when combined with ``method='ROC'``
            **kwargs: Other parameters specific to each fit and/or outlier
                screening method

        Returns:
            beta (numpy.ndarray): The array of regression estimators
            residuals (numpy.ndarray): The array of residuals

        Raises:
            NotImplementedError: If method is not yet implemented
            ValueError: Unknown value for `method`
        """
        numba.set_num_threads(n_threads)
        # Check for strictly increasing time dimension:
        if not np.all(dataarray.time.values[1:] >= dataarray.time.values[:-1]):
            raise ValueError("Time dimension of dataarray has to be sorted chronologically.")
        # lower level functions using numba may require that X and y have the same
        # datatype (e.g. float64, float64 signature)
        # If the precision is below float64, occurences of singular matrices get
        # more likely with short time series (i.e. especially for stable fits)
        y = dataarray.values.astype(np.float64)
        X = X.astype(np.float64)
        # If no mask has been set at class instantiation, assume everything is forest
        if self.mask is None:
            self.mask = np.ones_like(y[0,:,:], dtype=np.uint8)
        # Check if fit_start exists. If it does and is None, initialize it
        if getattr(self, 'fit_start', False) is None:
            start_date = dataarray.time.values.min() \
                .astype('datetime64[D]').astype('int')
            self.fit_start = np.full_like(self.mask, start_date, dtype=np.uint16)
        mask_bool = self.mask == 1
        shape = y.shape
        beta_shape = (X.shape[1], shape[1], shape[2])
        # Create empty arrays with output shapes to store reg coefficients and residuals
        beta = np.zeros(beta_shape, dtype=np.float32)
        residuals = np.zeros_like(y, dtype=np.float32)
        y_flat = y[:, mask_bool]
        y_flat = self._mask_short_series(y_flat, X)

        # 1. Optionally screen outliers
        #   This just updates y_flat
        if screen_outliers == 'Shewhart':
            y_flat = shewhart(X, y_flat, **kwargs)
            y_flat = self._mask_short_series(y_flat, X)
        elif screen_outliers == 'CCDC_RIRLS':
            try:
                green_flat = kwargs.pop('green').values\
                    .astype(np.float64)[:, self.mask == 1]
                swir_flat = kwargs.pop('swir').values\
                    .astype(np.float64)[:, self.mask == 1]
            except (KeyError, AttributeError):
                raise ValueError('green and swir xarray.Dataarray(s) need to be'
                                 ' provided using green and swir arguments'
                                 ' respectively')
            y_flat = ccdc_rirls(X, y_flat,
                                green=green_flat, swir=swir_flat, **kwargs)
            y_flat = self._mask_short_series(y_flat, X)
        elif screen_outliers:
            raise ValueError('Unknown screen_outliers')

        mask_bool = self.mask == 1

        # 2. Fit using specified method
        if method == 'ROC':
            # Convert datetime64 to days, so numba is happy
            dates = dataarray.time.values.astype('datetime64[D]').astype('int')
            # crit already calculated here, to allow numba in roc_stable_fit
            crit = _cusum_rec_test_crit(**kwargs)
            # Suppress numba np.dot() warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # ROC requires double precision when using numba
                beta_flat, residuals_flat, is_stable, fit_start = \
                    roc_stable_fit(X, y_flat, dates, crit=crit, **kwargs)
            self.mask.flat[np.flatnonzero(mask_bool)[~is_stable]] = 2
            if hasattr(self, 'fit_start'):
                self.fit_start[mask_bool] = fit_start
        elif method == 'CCDC-stable':
            if not self.trend:
                raise ValueError('Method "CCDC-stable" requires "trend" to be true.')
            dates = dataarray.time.values.astype('datetime64[D]').astype('int')
            beta_flat, residuals_flat, is_stable, fit_start = \
                ccdc_stable_fit(X, y_flat, dates, **kwargs)
            self.mask.flat[np.flatnonzero(mask_bool)[~is_stable]] = 2
            if hasattr(self, 'fit_start'):
                self.fit_start[mask_bool] = fit_start
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

    def monitor(self, array, date):
        """Monitor given a new acquisition

        The method takes care of (1) predicting the expected pixels values,
        (2) updating the process value, and (3) updating self.mask in case a
        break is confirmed

        Args:
            array (np.ndarray): 2D array containing the new acquisition to be
                monitored
            date (datetime.datetime): Date of acquisition of data contained in
                the array
        """
        if not isinstance(date, datetime.date):
            raise TypeError("'date' has to be of type datetime.date")
        if self.detection_date is None:
            self.detection_date = np.zeros_like(self.mask, dtype=np.uint16)
        y_pred = self.predict(date)
        residuals = array - y_pred
        # Compute a mask of values that can be worked on
        is_valid = np.logical_and(self.mask == 1, np.isfinite(array))
        is_valid = self._detect_extreme_outliers(residuals=residuals,
                                                 is_valid=is_valid)
        self._update_process(residuals=residuals, is_valid=is_valid)
        is_break = self._detect_break()
        # Update mask (3 value corresponds to a confirmed break)
        to_update = np.logical_and(is_valid, is_break)
        self.mask[to_update] = 3
        # Update detection date
        days_since_epoch = (date - datetime.datetime(1970, 1, 1)).days
        self.detection_date[to_update] = days_since_epoch

    def _detect_break(self):
        """Defines if the current process value is a confirmed break

        This method may be overridden in subclass if required
        """
        return np.abs(self.process) >= self.boundary

    def _detect_extreme_outliers(self, residuals, is_valid):
        """Detect extreme outliers in an array of residuals from prediction

        Sometimes used as pre-filtering of incoming new data to discard eventual
        remaining clouds for instance
        When implemented in a subclass this method must identify outliers and update
        the ``is_valid`` input array accordingly
        The base class provides a fallback that simply return the input ``is_valid``
        array
        """
        return is_valid

    @abc.abstractmethod
    def _update_process(self, residuals, is_valid):
        """Update process values given an array of residuals

        Args:
            residuals (np.ndarray): A 2D array of residuals
            is_valid (np.ndarray): A boolean 2D array indicating where process
                values should be updated
        """
        pass

    def _report(self, layers, dtype):
        """Prepare data to be written to disk by ``self.report``

        If overriden in subclass this method must generate a 3D numpy array (even
        when a single layer is returned) with geotiff compatible datatype. Axis order
        must be (band, y, x) as per the rasterio data model

        Args:
            layers (list): A list of strings indicating the layers to include in
                the report. Valid options are ``'mask'`` (the main output layer
                containing confirmed breaks, non-monitored pixels, etc), ``'detection_date'``
                (an integer value matching each confirmed break and indicating the date
                 the break was confirmed in days since epoch), ``'process'`` (the process
                 value). The process value has a different meaning and interpretation
                 for each monitoring method.
            dtype (type): The datatype of the stacked layers. Note that when returning
                process value for MoSum, CuSum or EWMA the ``dtype`` should be set
                to a float type to retain values

        Returns:
            numpy.ndarray: A 3D array with requested layers. Provided list order
                is respected
        """
        valid = ['mask', 'detection_date', 'process']
        if not all([x in valid for x in layers]):
            raise ValueError('invalid layer(s) requested')
        returned_layer = [getattr(self, x) for x in layers]
        return np.stack(returned_layer, axis=0).astype(dtype)

    def report(self, filename, layers=['mask', 'detection_date'],
               driver='GTiff', crs=CRS.from_epsg(3035),
               dtype=np.int16):
        """Write the result of reporting to a raster geospatial file

        Args:
            layers (list): A list of strings indicating the layers to include in
                the report. Valid options are ``'mask'`` (the main output layer
                containing confirmed breaks, non-monitored pixels, etc), ``'detection_date'``
                (an integer value matching each confirmed break and indicating the date
                 the break was confirmed in days since epoch), ``'process'`` (the process
                 value). The process value has a different meaning and interpretation
                 for each monitoring method.
            dtype (type): The datatype of the stacked layers. Note that when returning
                process value for MoSum, CuSum or EWMA the ``dtype`` should be set
                to a float type to retain values
        """
        r = self._report(layers=layers, dtype=dtype)
        count = r.shape[0]
        meta = {'driver': driver,
                'crs': crs,
                'count': count,
                'dtype': r.dtype,
                'transform': self.transform,
                'height': r.shape[-2],
                'width': r.shape[-1]}
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.descriptions = layers
            dst.write(r)

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
                # TODO A different way to name the third dimensions would be
                #  good. Right now the names might also clash with other
                #  attribute names (unlikely, but e.g. n, h in MOSUM)
                if k in src.dimensions.keys():
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
            # Create coordinate variables
            x_var = dst.createVariable('x', self.x.dtype, ('x',))
            y_var = dst.createVariable('y', self.y.dtype, ('y',))
            # fill values of coordinate variables
            x_var[:] = self.x
            y_var[:] = self.y

            # Starting letter for third dimensions
            third = 'a'
            for k,v in attr.items():
                if k not in ['x', 'y']:
                    if isinstance(v, np.ndarray):
                        if v.ndim == 3:
                            dim_3 = dst.createDimension(third, v.shape[0])
                            var_3 = dst.createVariable(third, np.uint16, (third,))
                            var_3[:] = np.arange(start=0,
                                                 stop=v.shape[0],
                                                 dtype=np.uint8)
                            var_3d = dst.createVariable(k, v.dtype,
                                                        (third, 'y', 'x'),
                                                        zlib=True)
                            var_3d[:] = v
                            third = chr(ord(third) + 1)
                            continue
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

    def _mask_short_series(self, y_flat, X):
        """ Masks short time series

        Time series shorter than 1.5x the number of regressors are masked
        and a warning is given.
        If after this no time series are left a ValueError is raised

        Args:
            y_flat (np.ndarray): 2D matrix of observations
            X (np.ndarray): 2D Matrix of regressors

        Returns:
            (np.ndarray) y_flat with short time-series removed
        """
        likely_singular = np.count_nonzero(~np.isnan(y_flat), axis=0) < (X.shape[1]*1.5)
        amount = np.count_nonzero(likely_singular)
        if amount:
            self.mask.flat[np.flatnonzero(self.mask == 1)[likely_singular]] = 4
            warnings.warn(f'{amount} time-series were shorter than 1.5x the '
                          f'number of regressors and were masked.')
        if not np.any(self.mask == 1):
            raise ValueError(f'There are no time-series with sufficient ({int(X.shape[1]*1.5)}) data points.')
        return y_flat[:,~likely_singular]
