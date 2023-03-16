# Copyright (C) 2022 European Union (Joint Research Centre)
#
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#   https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.

import numpy as np

from nrt.monitor import BaseNrt


class EWMA(BaseNrt):
    """Monitoring using EWMA control chart

    Implementation following method described in Brooks et al. 2014.

    Args:
        mask (numpy.ndarray): A 2D numpy array containing pixels that should be
            monitored marked as ``1`` and pixels that should be excluded (marked
            as ``0``). Typically a stable forest mask when doing forest disturbance
            monitoring. If no mask is supplied all pixels are considered and
            a mask is created following the ``fit()`` call
        trend (bool): Indicate whether stable period fit is performed with
            trend or not
        harmonic_order (int): The harmonic order of the time-series regression
        lambda_ (float): Weight of previous observation in the monitoring process
            (memory). Valid range is [0,1], 1 corresponding to no memory and 0 to
            full memory
        sensitivity (float): Sensitivity parameter used in the computation of the
            monitoring boundaries. Lower values imply more sensitive monitoring
        threshold_outlier (float): Values bigger than threshold_outlier*sigma
            (extreme outliers) will get screened out during monitoring and will
            not contribute to updating the EWMA process value
        **kwargs: Used to set internal attributes when initializing with
            ``.from_netcdf()``
    """
    def __init__(self, trend=True, harmonic_order=2, sensitivity=2, mask=None,
                 lambda_=0.3, threshold_outlier=10, **kwargs):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         **kwargs)
        self.lambda_ = lambda_
        self.sensitivity = sensitivity
        self.threshold = threshold_outlier
        self.sigma = kwargs.get('sigma')
        self.monitoring_strategy = 'EWMA'

    def fit(self, dataarray, method='OLS',
            screen_outliers='Shewhart', L=5, **kwargs):
        """Stable history model fitting

        The preferred fitting method for this monitoring type is ``'OLS'`` with
        outlier screening ``'Shewhart'``. It requires a control limit parameter
        ``L``. See ``nrt.outliers.shewart`` for more details
        """
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)
        beta, residuals = self._fit(X, dataarray=dataarray, method=method,
                                    screen_outliers=screen_outliers, L=L,
                                    **kwargs)
        self.beta = beta
        # get new standard deviation
        self.sigma = np.nanstd(residuals, axis=0)
        # calculate EWMA control limits and save them
        # since control limits quickly approach a limit they are assumed to be
        # stable after the training period and can thus be simplified
        self.boundary = self.sensitivity * self.sigma * np.sqrt((
            self.lambda_ / (2 - self.lambda_)))
        # calculate the EWMA value for the end of the training period and save it
        self.process = self._init_process(residuals)
        # Mark everything as unstable that already crosses the boundary after
        # fitting
        self.mask[self.process > self.boundary] = 2

    def _detect_extreme_outliers(self, residuals, is_valid):
        is_eoutlier = np.abs(residuals) > self.threshold * self.sigma
        return np.logical_and(~is_eoutlier, is_valid)

    def _update_process(self, residuals, is_valid):
        """Update process value (EWMA in this case) with new acquisition

        Args:
            residuals (numpy.ndarray): 2 dimensional array corresponding to the
                residuals of a new acquisition
            is_valid (np.ndarray): A boolean 2D array indicating where process
                values should be updated

        Returns:
            numpy.ndarray: A 2 dimensional array containing the updated EWMA
                values
        """
        # If the monitoring has not been initialized yet, raise an error
        if self.process is None:
            raise ValueError('Process has to be initialized before update')
        # Update ewma value for element of the input array that are not Nan
        process_new = self._update_ewma(array=residuals, ewma=self.process,
                                        lambda_=self.lambda_)
        self.process = np.where(is_valid, process_new, self.process)

    @staticmethod
    def _update_ewma(array, ewma, lambda_):
        ewma_new = np.where(np.isnan(array),
                            ewma,
                            (1 - lambda_) * ewma + lambda_ * array)
        return ewma_new

    def _init_process(self, array):
        """Initialize the ewma process value using the residuals of the fitted values

        Args:
            array (np.ndarray): 3 dimensional array of residuals. Usually the
                residuals from the model fitting

        Returns:
            numpy.ndarray: A 2 dimensional array corresponding to the last slice
            of the recursive ewma process updating
        """
        ewma = np.zeros_like(array[0,:,:])
        for slice_ in array:
            ewma = self._update_ewma(array=slice_, ewma=ewma, lambda_=self.lambda_)
        return ewma


