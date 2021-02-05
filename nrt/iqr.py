import numpy as np

from nrt import BaseNrt


class Iqr(BaseNrt):
    def __init__(self, mask=None, trend=True, harmonic_order=3, beta=None,
                 sensitivity=1.5, process=None, q25=None, q75=None,
                 boundary=3, x_coords=None, y_coords=None, **kwargs):
        """Online monitoring of disturbances based on interquartile range

        Reference:
            https://stats.stackexchange.com/a/1153
        """
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         beta=beta,
                         x_coords=x_coords,
                         y_coords=y_coords)
        self.monitoring_strategy = 'IQR'
        self.sensitivity = sensitivity
        self.process = process
        self.q25 = q25
        self.q75 = q75
        self.boundary = boundary

    def fit(self, dataarray, method='OLS', **kwargs):
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)
        beta, residuals = self._fit(X, dataarray=dataarray, method=method,
                                    **kwargs)
        self.beta = beta
        q75, q25 = np.nanpercentile(residuals, [75 ,25], 0)
        self.q25 = q25
        self.q75 = q75

    def _update_process(self, residuals, is_valid):
        # Compute upper and lower thresholds
        iqr = self.q75 - self.q25
        lower_limit = self.q25 - self.sensitivity * iqr
        upper_limit = self.q75 + self.sensitivity * iqr
        # compare residuals to thresholds
        is_outlier = np.logical_or(residuals > upper_limit,
                                   residuals < lower_limit)
        # Update self.process
        if self.process is None:
            self.process = np.zeros_like(residuals, dtype=np.uint8)
        self.process = np.where(is_valid,
                                self.process * is_outlier + is_outlier,
                                self.process)
