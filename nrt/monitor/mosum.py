import numpy as np
import xarray as xr

from nrt.monitor import BaseNrt
from nrt.utils_cusum import _mosum_ols_test_crit, _init_mosum_window


class MoSum(BaseNrt):
    """Monitoring using moving sums (MOSUM) of residuals

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
        sensitivity (float): sensitivity of the monitoring. Lower numbers
            correspond to lower sensitivity. Equivalent to significance level
            'alpha' with which the boundary is computed
        boundary (numpy.ndarray): process boundary for each time series.
            Calculated from alpha and length of time series.
        sigma (numpy.ndarray): Standard deviation for normalized residuals in
            history period
        histsize (numpy.ndarray): Number of non-nan observations in history
            period
        n (numpy.ndarray): Total number of non-nan observations in time-series
        critval (float): Critical test value corresponding to the chosen
            sensitivity

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
        sensitivity (float): sensitivity of the monitoring. Lower numbers
            correspond to lower sensitivity. Equivalent to significance level
            'alpha' with which the boundary is computed
        boundary (numpy.ndarray): process boundary for each time series.
            Calculated from alpha and length of time series.
        sigma (numpy.ndarray): Standard deviation for normalized residuals in
            history period
        histsize (numpy.ndarray): Number of non-nan observations in history
            period
        n (numpy.ndarray): Total number of non-nan observations in time-series
    """

    def __init__(self, mask=None, trend=True, harmonic_order=2, beta=None,
                 x_coords=None, y_coords=None, process=None, sensitivity=0.05,
                 boundary=None, sigma=None, histsize=None, n=None, h=0.25,
                 winsize=None, **kwargs):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         beta=beta,
                         x_coords=x_coords,
                         y_coords=y_coords,
                         process=process,
                         boundary=boundary)
        self.sensitivity = sensitivity
        self.critval = _mosum_ols_test_crit(sensitivity, win_size=h,
                                            period=10, functional='max')
        self.sigma = sigma
        self.histsize = histsize
        self.n = n
        self.h = h
        self.winsize = winsize

    def fit(self, dataarray, method='ROC', alpha=0.05, **kwargs):
        """Stable history model fitting

        The stability check will use the same sensitivity as is later used for
        detecting changes (default: 0.05)

        Args:
            dataarray (xr.DataArray): xarray Dataarray including the historic
                data to be fitted
            method (string): Regression to use. See ``_fit()`` for info.
            alpha (float): Significance level for ``'ROC'`` stable fit.
            **kwargs: to be passed to ``_fit``
        """
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)
        self.beta, residuals = self._fit(X, dataarray,
                                         method=method,
                                         alpha=alpha,
                                         **kwargs)
        # Flatten
        residuals_flat = residuals.reshape([len(residuals), -1])

        # histsize is necessary for normalization of residuals,
        # n is necessary for boundary calculation
        self.histsize = np.sum(~np.isnan(residuals_flat), axis=0) \
            .astype(np.uint16)
        self.winsize = np.floor(self.histsize * self.h).astype(np.int16)
        self.n = self.histsize
        self.boundary = np.full_like(self.histsize, np.nan, dtype=np.float32)
        self.sigma = np.nanstd(residuals_flat, axis=0, ddof=X.shape[1])
        # calculate normalized residuals
        residuals_ = residuals_flat / (self.sigma * np.sqrt(self.histsize))
        self.values = _init_mosum_window(residuals_, self.winsize)

    def get_process(self):
        return np.nansum(self.values, axis=0)

    def set_process(self, x):
        pass

    process = property(get_process, set_process)

    def _update_process(self, residuals, is_valid):
        # get valid idx
        is_valid = is_valid.ravel()
        valid_idx = np.where(is_valid)

        # get remainder, starting at 0 for first update
        remainder = np.mod(self.n-self.histsize, self.winsize)
        change_idx = (-self.winsize+remainder)[valid_idx]

        # normalize residuals
        residuals_norm = residuals.ravel() / (self.sigma * np.sqrt(self.histsize))

        # set new residuals at appropriate indices
        self.values[change_idx, valid_idx] = residuals_norm[valid_idx]

        # calculate boundary
        self.n = self.n + is_valid
        x = self.n / self.histsize
        log_out = np.ones_like(x)
        self.boundary = np.where(is_valid,
                                 self.critval * np.sqrt(
                                     2 * np.log(x, out=log_out,
                                                where=(x > np.exp(1)))),
                                 self.boundary)

    def _detect_break(self):
        """Defines if the current process value is a confirmed break"""
        is_break = np.abs(self.process) > self.boundary
        return is_break.reshape((self.beta.shape[1],
                                 self.beta.shape[2]))
