import numpy as np
import xarray as xr

from nrt.monitor import BaseNrt


class CCDC(BaseNrt):
    """Monitoring using CCDC-like implementation

    Implementation loosely following method described in Zhu & Woodcock 2014.

    Zhu, Zhe, and Curtis E. Woodcock. 2014. “Continuous Change Detection and
    Classification of Land Cover Using All Available Landsat Data.” Remote
    Sensing of Environment 144 (March): 152–71.
    https://doi.org/10.1016/j.rse.2014.01.011.

    Attributes:
        mask (numpy.ndarray): A 2D numpy array containing pixels that should
            be monitored (1) and not (0). The mask may be updated following
            history period stability check, and after a call to monitor
            following a confirmed break. Values are as follow.
            ``{0: 'Not monitored', 1: 'monitored', 2: 'Unstable history',
            3: 'Confirmed break - no longer monitored'}``
        trend (bool): Indicate whether stable period fit is performed with
            trend or not
        harmonic_order (int): The harmonic order of the time-series regression
        x (numpy.ndarray): array of x coordinates
        y (numpy.ndarray): array of y coordinates
        sensitivity (float): sensitivity of the monitoring. Lower numbers are
            high sensitivity. Value can't be zero.
        boundary (int): Number of consecutive observations identified as outliers
            to signal as disturbance
        nodata (np.ndarray): 2D Boolean array. Signals missing data in the newest
            acquisition
        rmse (np.ndarray): 2D float array indicating RMSE for each pixel

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
            high sensitivity. Value can't be zero.
        boundary (int): Number of consecutive observations identified as outliers
            to signal as disturbance
        rmse (np.ndarray): 2D float array indicating RMSE for each pixel
    """
    def __init__(self, mask=None, trend=True, harmonic_order=2, beta=None,
                 x_coords=None, y_coords=None, sensitivity=3, boundary=3,
                 process=None, rmse=None, detection_date=None, **kwargs):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         beta=beta,
                         x_coords=x_coords,
                         y_coords=y_coords,
                         detection_date=detection_date)
        self.sensitivity = sensitivity
        self.process = process
        self.boundary = boundary
        self.rmse = rmse

    def fit(self, dataarray, method='CCDC-stable', screen_outliers='CCDC_RIRLS',
            green=None, swir=None, scaling_factor=1, **kwargs):
        """Stable history model fitting

        If screen outliers is required, green and swir bands must be passed.

        The stability check will use the same sensitivity as is later used for
        detecting changes (default: 3*RMSE)

        Args:
            dataarray (xr.DataArray): xarray Dataarray including the historic
                data to be fitted
            method (string): Regression to use. See ``_fit()`` for info.
            screen_outliers (string): Outlier screening to use.
                See ``_fit()`` for info.
            green (xr.DataArray): Green reflectance values to be used by
                ``screen_outliers``.
            swir (xr.DataArray): Short wave infrared (SWIR) reflectance values
                to be used by ``screen_outliers``.
            scaling_factor (int): Optional Scaling factor to be applied to
                ``green`` and ``swir``.
            **kwargs: to be passed to ``_fit``
        """
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)
        self.beta, residuals = self._fit(X, dataarray,
                                         method=method,
                                         screen_outliers=screen_outliers,
                                         green=green, swir=swir,
                                         scaling_factor=scaling_factor,
                                         **kwargs)
        self.rmse = np.sqrt(np.nanmean(residuals ** 2, axis=0))

    def _update_process(self, residuals, is_valid):
        # TODO: Calculation is different for multivariate analysis
        # (mean of all bands has to be > sensitivity)
        is_outlier = np.abs(residuals) / self.rmse > self.sensitivity
        # Update process
        if self.process is None:
            self.process = np.zeros_like(residuals, dtype=np.uint8)
        self.process = np.where(is_valid,
                                self.process * is_outlier + is_outlier,
                                self.process)
