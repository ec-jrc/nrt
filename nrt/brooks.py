import numpy as np
import xarray as xr
from nrt import BaseNrt


class Brooks(BaseNrt):
    def __init__(self, mask=None, trend=True, harmonic_order=2, beta=None, x_coords=None, y_coords=None,
                 sensitivity=0.3, threshold=2, sigma=None, cl_ewma=None, ewma=None, nodata=None, **kwargs):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         beta=beta,
                         x_coords=x_coords,
                         y_coords=y_coords)
        self.sensitivity = sensitivity
        self.threshold = threshold
        self.sigma = sigma
        self.cl_ewma = cl_ewma  # control limit
        self.ewma = ewma  # array with most recent EWMA values
        self.nodata = nodata # Only necessary for singalling missing data
        self.test = np.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64[D]')

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
        dataarray.values[shewhart_mask] = np.nan

        # fit again, but without outliers
        beta, residuals = self._fit(X, dataarray=dataarray, reg=reg,
                                    check_stability=check_stability,
                                    **kwargs)
        self.beta = beta
        self.nodata = np.isnan(residuals[-1])

        # get new standard deviation
        self.sigma = np.nanstd(residuals, axis=0)

        # calculate EWMA control limits and save them
        # since control limits quickly approach a limit they are assumed to be stable after the training period
        # and can thus be simplified
        self.cl_ewma = self.threshold * self.sigma * np.sqrt((self.sensitivity / (2 - self.sensitivity)))

        # calculate the EWMA value for the end of the training period and save it
        self.ewma = self.calc_ewma(residuals, sensitivity=self.sensitivity)[-1]

    def monitor(self, array, date):
        y_pred = self.predict(date)
        residuals = array - y_pred

        # TODO EWMA calculation in fit and monitor by calc_ewma()
        # Filtering of values with high threshold X-Bar and calculating new EWMA values
        residuals[np.abs(residuals) > self.threshold * self.sigma] = np.nan
        self.nodata = np.isnan(residuals[-1])

        self.ewma = np.where(np.isnan(residuals),
            self.ewma,
            (1 - self.sensitivity) * self.ewma + self.sensitivity * residuals)

    def _report(self):
        # signals severity of disturbance:
        #    0 = not disturbed
        #   >1 = disturbed (bigger number: higher severity)
        #  255 = no data
        # TODO no data signal works, but is ugly, maybe make it optional
        signal = np.floor_divide(np.abs(self.ewma), self.cl_ewma).astype(np.uint8)
        signal[self.nodata] = 255
        return signal

    # TODO Check if Numba works
    # @numba.jit("float32[:](float32[:], float32[:])", nopython=True, nogil=True)
    @staticmethod
    def calc_ewma(residuals, ewma=0, sensitivity=0.3):
        """ Calculates EWMA for every value in residuals

            Args:
                residuals (numpy.ndarray): 2 or 3 dimensional array of residuals
                ewma (numpy.ndarray): 2 dimensional array of previous EWMA values
                sensitivity (float): Sensitivity of the EWMA chart to previous values (0: high, 1: low)
            Returns:
                numpy.ndarray: EWMA values in the same format as residuals
            """
        ewma_new = np.empty(np.shape(residuals), dtype=np.float32)

        # TODO handling of 2D array could probably be nicer
        is_2d = residuals.ndim == 2
        if is_2d:
            ewma_new = np.array([ewma_new])

        # initialize ewma with 0
        ewma_new[0] = np.where(np.isnan(residuals[0]),
                               ewma,
                               (1 - sensitivity) * ewma + sensitivity * residuals[0])
        for i in range(1, len(residuals)):
            ewma_new[i] = np.where(np.isnan(residuals[i]),
                               ewma_new[i - 1],
                               (1 - sensitivity) * ewma_new[i - 1] + sensitivity * residuals[i])
        if is_2d:
            ewma_new = ewma_new[0]
        return ewma_new
