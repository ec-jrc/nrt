import numpy as np

from nrt import BaseNrt


class Iqr(BaseNrt):
    def __init__(self, mask=None, trend=True, harmonic_order=3, beta=None,
                 sensitivity=1.5, state=None, q25=None, q75=None,
                 confirm_threshold=3, x_coords=None, y_coords=None, **kwargs):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         beta=beta,
                         x_coords=x_coords,
                         y_coords=y_coords)
        self.monitoring_strategy = 'IQR'
        self.sensitivity = sensitivity
        self.state = state
        self.q25 = q25
        self.q75 = q75
        self.confirm_threshold = confirm_threshold

    def fit(self, dataarray, reg='OLS', check_stability=None, **kwargs):
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)
        beta, residuals = self._fit(X, dataarray=dataarray, method=reg,
                                    check_stability=check_stability,
                                    **kwargs)
        self.beta = beta
        q75, q25 = np.nanpercentile(residuals, [75 ,25], 0)
        self.q25 = q25
        self.q75 = q75

    def monitor(self, array, date):
        iqr = self.q75 - self.q25
        lower_limit = self.q25 - self.sensitivity * iqr
        upper_limit = self.q75 + self.sensitivity * iqr
        y_pred = self.predict(date)
        # Compute residuals
        residuals = array - y_pred
        # compare residuals to threshold
        is_outlier = np.logical_or(residuals > upper_limit, residuals < lower_limit)
        # Update self.state
        if self.state is None:
            self.state = np.zeros_like(array, dtype=np.uint8)
        # TODO: Pixels that could not be monitored (because cloud/Nan) should not be left_shifted
        # see where argument of np.left_shift
        self.state = np.left_shift(self.state, np.uint(1))
        self.state = self.state + is_outlier
        # TODO: update mask (confirmed breaks are no longer monitored)

    def _report(self):
        # TODO: add different levels of flagging (level 1, 2, 3, confirmed, ...)
        # TODO: Merge with updated mask to show unmonitored and previously disturbed areas
        bit_mask = sum([2**x for x in range(self.confirm_threshold)])
        flags = np.where(np.bitwise_and(self.state, bit_mask) == bit_mask, 1, 0)
        return flags.astype(np.uint8)
