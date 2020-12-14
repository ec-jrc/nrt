import abc

import numpy as np
import pandas as pd

from nrt.utils import build_regressors

__version__ = "0.0.1"


class BaseNrt(metaclass=abc.ABCMeta):
    def __init__(self, mask=None, trend=True, harmonic_order=3):
        self.mask = mask
        self.trend = trend
        self.harmonic_order = harmonic_order

    def _fit(self, X, dataarray, reg='OLS', check_stability=None, **kwargs):
        """

        Args:
            dataarray (xarray.DataArray): A 3 dimension DataArray
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
        shape_beta = self.beta.shape
        shape_y = (shape_beta[1], shape_beta[2])
        shape_beta_flat = (shape_beta[0], shape_beta[1] * shape_beta[2])
        X = self.regressors(date)
        y_pred = np.dot(X, self.beta.reshape(shape_beta_flat))
        return y_pred.reshape(shape_y)

    @classmethod
    def from_netcdf(cls, filename, **kwargs):
        pass

    def to_netcdf(self):
        pass

    @staticmethod
    def build_design_matrix(dataarray, trend=True, harmonic_order=3):
        dates = pd.DatetimeIndex(dataarray.time.values)
        X = build_regressors(dates=dates, trend=trend, harmonic_order=harmonic_order)
        return X

    def regressors(self, date):
        date_pd = pd.DatetimeIndex([date])
        X = build_regressors(dates=date_pd, trend=self.trend,
                             harmonic_order=self.harmonic_order)
        return X


