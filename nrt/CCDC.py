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
                 x_coords=None, y_coords=None, sensitivity=3, **kwargs):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         beta=beta,
                         x_coords=x_coords,
                         y_coords=y_coords)
        self.sensitivity = sensitivity

    def fit(self, dataarray, reg='ols', screen_outliers='CCDC_RIRLS',
            check_stability='CCDC', green=None, swir=None, **kwargs):
        """Stable history model fitting

        Much more complicated call than for shewhart. If screen outliers is
        required, green and swir bands must be passed.

        The stability check will use the same sensitivity as is later used for
        detecting changes (default: 3*RMSE)
        """
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)

        beta, residuals = self._fit(X, dataarray,
                                    method='OLS',
                                    screen_outliers='CCDC_RIRLS',
                                    check_stability='CCDC',
                                    green=green, swir=swir)

    def monitor(self, array, date):
        y_pred = self.predict(date)
        residuals = array - y_pred
        # Filtering of values with high threshold X-Bar and calculating new EWMA values
        residuals[np.abs(residuals) > self.sensitivity * self.sigma] = np.nan
        self.nodata = np.isnan(residuals)
        self.process = self._update_process(array=residuals)

    def _report(self):
        # signals severity of disturbance:
        #    0 = not disturbed
        #   >1 = disturbed (bigger number: higher severity)
        #  255 = no data
        # TODO no data signal works, but is ugly, maybe make it optional
        signal = np.floor_divide(np.abs(self.process),
                                 self.boundary).astype(np.uint8)
        signal[self.nodata] = 255
        return signal