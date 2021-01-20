import numpy as np
import xarray as xr
import numba

from nrt import BaseNrt


class CCDC(BaseNrt):
    """Monitoring using CCDC-like implementation

    Implementation loosely following method described in Zhu & Woodcock 2014.

    Args:
        lambda_ (float): Weight of previous observation in the monitoring process
            (memory). Valid range is [0,1], 1 corresponding to no memory and 0 to
            full memory
        sensitivity (float): Sensitivity parameter used in the computation of the
            monitoring boundaries. Lower values imply more sensitive monitoring
    """
    def __init__(self, mask=None, trend=True, harmonic_order=2, beta=None,
                 x_coords=None, y_coords=None, sensitivity=2, **kwargs):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         beta=beta,
                         x_coords=x_coords,
                         y_coords=y_coords)
        self.sensitivity = sensitivity

    def fit(self, dataarray, reg='ols', screen_outliers='rirls',
            check_stability='ccdc', green=None, swir=None, **kwargs):
        """Stable history model fitting

        Much more complicated call than for shewhart. If screen outliers is
        required, green, swir and the threshold for screening must be passed.

        The stability check will use the same sensitivity as is later used for
        detecting changes (default: 3*RMSE)
        """
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)

        # 1. Screen outliers

        # 2. Fit "Normal" regression

        # 3. Check stability of 2.

        # 3.a unstable: change the time frame (how?), refit and check
        #               stability again.
        #               If after the smallest possible time frame (365 days)
        #               it is not stable -> won't be monitored

        # 3.b stable: Done

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