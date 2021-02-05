import numpy as np
import xarray as xr
import numba

from nrt import BaseNrt


class EWMA(BaseNrt):
    """Monitoring using EWMA control chart

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

    def fit(self, dataarray, method='OLS',
            screen_outliers='Shewhart', **kwargs):
        """Stable history model fitting

        The preferred fitting method for this monitoring type is ``'Shewhart'``.
        It requires a control limit parameter ``L``. See ``nrt.fit_methods.shewart``
        for more details
        """
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)
        beta, residuals = self._fit(X, dataarray=dataarray,
                                    method=method, **kwargs)
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
        self.process = self._init_process(residuals)

    def _detect_extreme_outliers(self, residuals, is_valid):
        is_eoutlier = np.abs(residuals) > self.sensitivity * self.sigma
        return np.logical_and(~is_eoutlier, is_valid)

    def _update_process(self, residuals, is_valid):
        """Update process value (EWMA in this case) with new acquisition

        Args:
            residuals (numpy.ndarray): 2 dimensional array corresponding to the residuals
                of a new acquisition
            is_valid (np.ndarray): A boolean 2D array indicating where process
                values should be updated

        Returns:
            numpy.ndarray: A 2 dimensional array containing the updated EWMA values
        """
        # If the monitoring has not been initialized yet, raise an error
        if self.process is None:
            raise ValueError('Process has to be initialized before update')
        # Update ewma value for element of the input array that are not Nan
        process_new = self._update_ewma(array=residuals, ewma=self.process,
                                        lambda_=self.lambda_)
        self.process = np.where(is_valid, process_new, self.process)

    def _detect_break(self):
        # TODO override _report() to also signal severity of disturbance
        is_break = np.floor_divide(np.abs(self.process),
                                   self.boundary).astype(np.uint8)
        return is_break

    @staticmethod
    def _update_ewma(array, ewma, lambda_):
        ewma_new = np.where(np.isnan(array),
                            ewma,
                            (1 - lambda_) * ewma + lambda_ * array)
        return ewma_new

    # TODO: only static methods can be jitted, make _update_ewma external to the class
    # and _init_process static
    def _init_process(self, array):
        """Initialize the ewma process value using the residuals of the fitted values

        Args:
            array (np.ndarray): 3 dimensional array of residuals. Usually the residuals
                from the model fitting

        Returns:
            numpy.ndarray: A 2 dimensional array corresponding to the last slice
            of the recursive ewma process updating
        """
        ewma = np.zeros_like(array[0,:,:])
        for slice_ in array:
            ewma = self._update_ewma(array=slice_, ewma=ewma, lambda_=self.lambda_)
        return ewma


