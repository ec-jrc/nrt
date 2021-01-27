import numpy as np
import xarray as xr

from nrt import BaseNrt

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
        nodata (np.ndarray): 2D Boolean array. Signals missing data in the newest
            acquisition
        rmse (np.ndarray): 2D float array indicating RMSE for each pixel
    """
    def __init__(self, mask=None, trend=True, harmonic_order=2, beta=None,
                 x_coords=None, y_coords=None, sensitivity=3,
                 boundary=3, process=None, nodata=None, rmse=None,
                 **kwargs):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         beta=beta,
                         x_coords=x_coords,
                         y_coords=y_coords)
        self.sensitivity = sensitivity
        self.process = process
        self.boundary = boundary
        self.nodata = nodata
        self.rmse = rmse

    def fit(self, dataarray, reg='ols', screen_outliers='CCDC_RIRLS',
            check_stability='CCDC', green=None, swir=None, scaling_factor=1,
            **kwargs):
        """Stable history model fitting

        If screen outliers is required, green and swir bands must be passed.

        The stability check will use the same sensitivity as is later used for
        detecting changes (default: 3*RMSE)

        Args:
            dataarray (xr.DataArray): xarray Dataarray including the historic
                data to be fitted
            reg (string): Regression to use. See ``_fit()`` for info.
            screen_outliers (string): Outlier screening to use.
                See ``_fit()`` for info.
            check_stability (string): Stability check to use.
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
                                         method=reg,
                                         screen_outliers=screen_outliers,
                                         check_stability=check_stability,
                                         green=green, swir=swir,
                                         scaling_factor=scaling_factor,
                                         **kwargs)
        self.rmse = np.sqrt(np.nanmean(residuals ** 2, axis=0))

    def monitor(self, array, date):
        """ Monitoring of forest disturbance

        Args:
            array (np.ndarray): 2D numpy array in the same format as used in
                ``fit()``
            date (np.datetime64): Date of the array
        """
        # TODO masking needs to be done in predict()
        y_pred = self.predict(date)
        residuals = array - y_pred
        self.nodata = np.isnan(residuals)
        # TODO: Calculation is different for multivariate analysis
        # (mean of all bands has to be > sensitivity)
        is_outlier = np.abs(residuals) / self.rmse > self.sensitivity
        # Update process
        if self.process is None:
            self.process = np.zeros_like(array, dtype=np.uint8)
        self.process = self.process * is_outlier + is_outlier

    def _report(self):
        # signals severity of disturbance:
        #    0 = not disturbed
        #   >1 = disturbed (bigger number: longer duration)
        #  255 = no data
        # TODO when masking is implemented in monitor(), change the reporting
        #   here as well
        signal = np.floor_divide(self.process,
                                 self.boundary).astype(np.uint8)
        signal[self.nodata] = 255
        return signal
