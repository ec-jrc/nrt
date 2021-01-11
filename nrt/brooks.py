import numpy as np

from nrt import BaseNrt


class Brooks(BaseNrt):
    def __init__(self, mask=None, trend=True, harmonic_order=2, beta=None, x_coords=None, y_coords=None,
                 sensitivity=0.3, threshold=2):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         beta=beta,
                         x_coords=x_coords,
                         y_coords=y_coords)
        self.sensitivity = sensitivity
        self.threshold = threshold

    def fit(self, dataarray, reg='OLS', check_stability=None, **kwargs):
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)
        beta, residuals = self._fit(X, dataarray=dataarray, reg=reg,
                                    check_stability=check_stability,
                                    **kwargs)
        self.beta = beta

        # Shewhart chart to get rid of outliers (clouds etc)
        sigma = np.nanstd(residuals, axis=2)
        shewhart_mask = np.abs(residuals) > self.threshold * sigma
        masked_outliers = np.where(shewhart_mask, np.nan, dataarray)

        # fit again, but without outliers
        beta, residuals = self._fit(X, dataarray=masked_outliers, reg=reg,
                                    check_stability=check_stability,
                                    **kwargs)
        self.beta = beta

        # calculate residuals including outliers and get standard deviation
        # TODO is it possible to not have to predict for everything but only for the masked values
        #  which aren't in `residuals`?
        y_pred = self.predict(dataarray.time.values)
        resid = dataarray - y_pred
        sigma = np.nanstd(resid, axis=2)

        # screen outliers for the last time
        shewhart_mask = np.abs(residuals) > self.threshold * sigma

        # calculate EWMA control limits and save them

    def monitor(self):