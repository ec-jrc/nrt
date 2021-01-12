import numpy as np
import xarray as xr
from nrt import BaseNrt


class Brooks(BaseNrt):
    def __init__(self, mask=None, trend=True, harmonic_order=2, beta=None, x_coords=None, y_coords=None,
                 sensitivity=0.3, threshold=2, sigma=None):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         beta=beta,
                         x_coords=x_coords,
                         y_coords=y_coords)
        self.sensitivity = sensitivity
        self.threshold = threshold
        self.sigma = sigma
        self.cl_ewma = None  # control limit
        self.ewma = None  # array with most recent EWMA values

    def fit(self, dataarray, reg='OLS', check_stability=None, **kwargs):
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)
        beta_full, residuals_full = self._fit(X, dataarray=dataarray, reg=reg,
                                    check_stability=check_stability,
                                    **kwargs)

        # Shewhart chart to get rid of outliers (clouds etc)
        sigma = np.nanstd(residuals_full, axis=0)
        shewhart_mask = np.abs(residuals_full) > (self.threshold * sigma)
        masked_outliers = dataarray.copy()
        masked_outliers.values = np.where(shewhart_mask, np.nan, dataarray)

        # fit again, but without outliers
        beta, residuals = self._fit(X, dataarray=masked_outliers, reg=reg,
                                    check_stability=check_stability,
                                    **kwargs)
        self.beta = beta

        # get new standard deviation
        self.sigma = np.nanstd(residuals, axis=0)

        # calculate EWMA control limits and save them
        # since control limits quickly approach a limit they are assumed to be stable after the training period
        # and can thus be simplified
        self.cl_ewma = self.threshold * self.sigma * np.sqrt((self.sensitivity / (2 - self.sensitivity)))

        # calculate the EWMA value for the end of the training period and save it
        self.ewma = self.calc_ewma(residuals)[-1]

    def monitor(self, array, date):
        y_pred = self.predict(date)
        residuals = array - y_pred

        # Filtering of values with high threshold X-Bar and calculating new EWMA values
        residuals[np.abs(residuals) > self.threshold * self.sigma] = np.nan

        self.ewma = np.where(np.isnan(residuals),
            self.ewma,
            (1 - self.sensitivity) * self.ewma + self.sensitivity * residuals)

    def _report(self):
        # signals severity of disturbance:
        #   0 = not disturbed
        #   >1 = disturbed
        # TODO: Signal clouds as 255 or something.
        return np.floor_divide(np.abs(self.ewma), self.cl_ewma).astype(np.uint8)

    # TODO Check if Numba works
    # @numba.jit("float32[:](float32[:], float32[:])", nopython=True, nogil=True)
    def calc_ewma(self, residuals):
        """ Calculates EWMA for every value in residuals

            Args:
                residuals (numpy.ndarray): 3 dimensional array of residuals

            Returns:
                numpy.ndarray: EWMA values
            """
        ewma = np.empty(np.shape(residuals), dtype=np.float32)
        # initialize ewma with 0
        ewma[0] = 0
        for i in range(1, len(residuals)):
            ewma[i] = np.where(np.isnan(residuals[i]),
                               ewma[i - 1],
                               (1 - self.sensitivity) * ewma[i - 1] + self.sensitivity * residuals[i])
        return ewma
