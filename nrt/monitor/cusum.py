import numpy as np
import xarray as xr

from nrt.monitor import BaseNrt
from nrt.utils_cusum import _cusum_ols_test_crit


class CuSum(BaseNrt):
    """Monitoring using cumulative sums (CUSUM) of residuals

    Implementation following method as implemented in R package bFast.

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
        sensitivity (float): sensitivity of the monitoring. Lower numbers are
            lower sensitivity. Corresponds to significance level alpha with
            which the boundary is computed. Value need to be between 0 and 1

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
        sensitivity (float): sensitivity of the monitoring. Lower numbers are
            lower sensitivity. Corresponds to significance level alpha with
            which the boundary is computed. Value need to be between 0 and 1
    """
    def __init__(self, mask=None, trend=True, harmonic_order=2, beta=None,
                 x_coords=None, y_coords=None, process=None, sensitivity=0.05,
                 boundary=None, sigma=None, histsize=None, n=None, **kwargs):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         beta=beta,
                         x_coords=x_coords,
                         y_coords=y_coords,
                         process=process,
                         boundary=boundary)
        self.sensitivity = sensitivity
        self.critval = _cusum_ols_test_crit(sensitivity)
        self.sigma = sigma
        self.histsize = histsize
        self.n = n

    def fit(self, dataarray, method='ROC', **kwargs):
        """Stable history model fitting

        The stability check will use the same sensitivity as is later used for
        detecting changes (default: 0.05)

        Args:
            dataarray (xr.DataArray): xarray Dataarray including the historic
                data to be fitted
            method (string): Regression to use. See ``_fit()`` for info.
            **kwargs: to be passed to ``_fit``
        """
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)
        self.beta, residuals = self._fit(X, dataarray,
                                         method=method,
                                         alpha=self.sensitivity,
                                         **kwargs)

        # histsize is necessary for normalization of residuals,
        # n is necessary for boundary calculation
        self.histsize = np.count_nonzero(~np.isnan(residuals), axis=0)
        self.n = self.histsize
        self.boundary = np.full_like(self.histsize, np.nan, dtype=np.uint16)
        # TODO X.shape might have to be subtracted by 1 because of intercept
        self.sigma = np.nanstd(residuals, axis=0, ddof=X.shape[1])
        # calculate process and normalize it using sigma and histsize
        process_ = np.nancumsum(residuals, axis=0)[-1]
        self.process = process_ / (self.sigma*np.sqrt(self.histsize))

    def _update_process(self, residuals, is_valid):
        # calculate boundary
        self.n = self.n + is_valid
        x = self.n / self.histsize
        # TODO: if n wasn't incremented and so x = 1, boundary calculation will
        #   return a warning, shouldn't change anything though.
        self.boundary = np.where(is_valid,
                                 np.sqrt(x * (x - 1)
                                    * (self.critval**2 + np.log(x / (x - 1)))),
                                 self.boundary)
        # normalize residuals
        residuals_norm = residuals / (self.sigma*np.sqrt(self.histsize))
        # Update process
        self.process = np.where(is_valid,
                                self.process+residuals_norm,
                                self.process)
