import numpy as np

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

        # calculate residuals including outliers and get standard deviation to get accurate base values
        # for monitoring
        # TODO is it possible to not have to predict for everything but only for the masked values
        #  which aren't in `residuals`?
        y_pred = self.predict(dataarray.time.values)
        resid = dataarray - y_pred
        sigma = np.nanstd(resid, axis=2)

        # screen outliers for the last time and calculate sigma to make monitoring more sensitive
        shewhart_mask = np.abs(resid) > self.threshold * sigma
        masked_outliers = np.where(shewhart_mask, np.nan, dataarray)
        self.sigma = np.nanstd(masked_outliers, axis=2)

        # calculate EWMA control limits and save them
        # since control limits quickly approach a limit they are assumed to be stable after the training period
        # and can thus be simplified
        self.cl_ewma = self.threshold * sigma * np.sqrt((self.sensitivity / 2 - self.sensitivity))

        # calculate the EWMA value for the end of the training period and save it
        self.ewma = self.calc_ewma(masked_outliers)[-1]

    def monitor(self, dataarray):
        y_pred = self.predict(dataarray.time.values)
        residuals = dataarray - y_pred

        # Filtering of values with high threshold X-Bar and calculating new EWMA values
        ewma = np.where(np.abs(residuals > self.threshold * self.sigma),
                        self.ewma,
                        (1 - self.sensitivity) * self.ewma + self.sensitivity * residuals)

        # 2.

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
