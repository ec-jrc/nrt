import abc

import numpy as np
import pandas as pd

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

    Args:
        mask (numpy.ndarray): A 2D numpy array containing pixels that should be
            monitored marked as ``1`` and pixels that should be excluded (marked
            as ``0``). Typically a stable forest mask when doing forest disturbance
            monitoring. If no mask is supplied all pixels are considered and
            a mask is created following the ``fit()`` call
        trend (bool): Indicate whether stable period fit is performed with
            trend or not
        harmonic_order (int): The harmonic order of the time-series regression
    """
    def __init__(self, mask=None, trend=True, harmonic_order=3):
        self.mask = mask
        self.trend = trend
        self.harmonic_order = harmonic_order

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
    def report(self):
        pass

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
        pass

    def to_netcdf(self):
        pass

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


