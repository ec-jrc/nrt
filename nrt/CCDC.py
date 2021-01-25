import numpy as np
import xarray as xr
import numba

from nrt import BaseNrt


class CCDC(BaseNrt):
    """Monitoring using CCDC-like implementation

    Implementation loosely following method described in Zhu & Woodcock 2014.

    Args:
        sensitivity (float): Sensitivity parameter used in the computation of the
            monitoring boundaries. Lower values imply more sensitive monitoring
    """

    def __init__(self, mask=None, trend=True, harmonic_order=2, beta=None,
                 x_coords=None, y_coords=None, sensitivity=3, scaling_factor=1,
                 boundary=3, process=None, nodata=None, rmse=None,
                 **kwargs):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         beta=beta,
                         x_coords=x_coords,
                         y_coords=y_coords)
        self.sensitivity = sensitivity
        self.scaling_factor = scaling_factor
        self.process = process
        self.boundary = boundary
        self.nodata = nodata
        self.rmse = rmse

    def fit(self, dataarray, reg='ols', screen_outliers='CCDC_RIRLS',
            check_stability='CCDC', green=None, swir=None, **kwargs):
        """Stable history model fitting

        If screen outliers is
        required, green and swir bands must be passed.

        The stability check will use the same sensitivity as is later used for
        detecting changes (default: 3*RMSE)
        """
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)

        self.beta, residuals = self._fit(X, dataarray,
                                         method=reg,
                                         screen_outliers=screen_outliers,
                                         check_stability=check_stability,
                                         green=green, swir=swir,
                                         scaling_factor=self.scaling_factor)
        self.rmse = np.sqrt(np.nanmean(residuals ** 2, axis=0))

    def monitor(self, array, date):
        # TODO masking needs to be done in predict()
        y_pred = self.predict(date)
        residuals = array - y_pred
        self.nodata = np.isnan(residuals)

        # Calculation is different for multivariate analysis
        # (mean of all bands has to be > sensitivity)
        is_outlier = np.abs(residuals) / self.rmse > self.sensitivity

        if self.process is None:
            self.process = np.zeros_like(array, dtype=np.uint8)
        self.process = self.process * is_outlier + is_outlier

    def _report(self):
        # signals severity of disturbance:
        #    0 = not disturbed
        #   >1 = disturbed (bigger number: higher severity)
        #  255 = no data
        # TODO no data signal works, but is ugly, maybe make it optional
        signal = np.floor_divide(self.process,
                                 self.boundary).astype(np.uint8)
        signal[self.nodata] = 255
        return signal
