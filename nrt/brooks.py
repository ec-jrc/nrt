import numpy as np
import xarray as xr
from nrt import BaseNrt


class Brooks(BaseNrt):
    """Monitoring using ewma control chart

    Implementation following method described in Brooks et al. 2014.

    Args:
        lambda_ (float): Weight of previous observation in the monitoring process
            (memory). Valid range is [0,1], 1 corresponding to no memory and 0 to
            full memory
        sensitivity (float): Sensitivity parameter used in the computation of the
            monitoring boundaries. Lower values imply more sensitive monitoring
    """
    def __init__(self, mask=None, trend=True, harmonic_order=2, beta=None,
                 x_coords=None, y_coords=None, lambda_=0.3, sensitivity=2,
                 sigma=None, boundary=None, process=None, nodata=None, **kwargs):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         beta=beta,
                         x_coords=x_coords,
                         y_coords=y_coords)
        self.lambda_ = lambda_
        self.sensitivity = sensitivity
        self.sigma = sigma
        self.boundary = boundary  # control limit
        self.process = process  # array with most recent EWMA values
        self.nodata = nodata  # Only necessary for singalling missing data

    def fit(self, dataarray, method='Shewhart', check_stability=None, **kwargs):
        """Stable history model fitting

        The preferred fitting method for this monitoring type is ``'Shewhart'``.
        It requires a control limit parameter ``L``. See ``nrt.fit_methods.shewart``
        for more details
        """
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)
        beta, residuals = self._fit(X, dataarray=dataarray, method=method,
                                    check_stability=check_stability, **kwargs)
        self.beta = beta
        self.nodata = np.isnan(residuals[-1])
        # get new standard deviation
        self.sigma = np.nanstd(residuals, axis=0)
        # calculate EWMA control limits and save them
        # since control limits quickly approach a limit they are assumed to be
        # stable after the training period and can thus be simplified
        self.boundary = self.sensitivity * self.sigma * np.sqrt((
            self.lambda_ / (2 - self.lambda_)))
        # calculate the EWMA value for the end of the training period and save it
        self.process = self.calc_ewma(residuals, lambda_=self.lambda_, ewma=0)[-1]

    def monitor(self, array, date):
        y_pred = self.predict(date)
        residuals = array - y_pred
        # TODO EWMA calculation in fit and monitor by calc_ewma()
        # Filtering of values with high threshold X-Bar and calculating new EWMA values
        residuals[np.abs(residuals) > self.sensitivity * self.sigma] = np.nan
        self.nodata = np.isnan(residuals)
        self.process = self.calc_ewma(residuals, lambda_=self.lambda_, ewma=self.process)[-1]

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

    # TODO Check if Numba works
    # @numba.jit("float32[:](float32[:], float32[:])", nopython=True, nogil=True)
    @staticmethod
    def calc_ewma(residuals, ewma=0, lambda_=0.3):
        """ Calculates EWMA for every value in residuals

            Args:
                residuals (numpy.ndarray): 2 or 3 dimensional array of residuals
                ewma (numpy.ndarray): 2 dimensional array of previous EWMA values
                lambda_ (float): Weight of previous values in the EWMA chart
                    (0: high, 1: low)
            Returns:
                numpy.ndarray: 3 dimensional array of EWMA values (even if residuals is 2D)
            """
        # TODO handling of 2D array could probably be nicer
        if residuals.ndim == 2:
            residuals = np.array([residuals])
        ewma_new = np.empty(np.shape(residuals), dtype=np.float32)
        # initialize ewma with 0
        ewma_new[0] = np.where(np.isnan(residuals[0]),
                               ewma,
                               (1 - lambda_) * ewma + lambda_ * residuals[0])
        for i in range(1, len(residuals)):
            ewma_new[i] = np.where(np.isnan(residuals[i]),
                                   ewma_new[i - 1],
                                   (1 - lambda_) * ewma_new[i - 1] + lambda_ * residuals[i])
        return ewma_new
