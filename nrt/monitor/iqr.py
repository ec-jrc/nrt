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
from nrt.stats import nan_percentile_axis0


class IQR(BaseNrt):
    """Online monitoring of disturbances based on interquartile range

        Reference:
            https://stats.stackexchange.com/a/1153

    Attributes:
        mask (numpy.ndarray): A 2D numpy array containing pixels that should
            be monitored (1) and not (0). The mask may be updated following
            history period stability check, and after a call to monitor
            following a confirmed break. Values are as follow.
            ``{0: 'Not monitored', 1: 'monitored', 2: 'Unstable history',
            3: 'Confirmed break - no longer monitored'}``
        trend (bool): Indicate whether stable period fit is performed with
            trend or not
        harmonic_order (int): The harmonic order of the time-series regression
        beta (np.ndarray): 3D array containing the model coefficients
        x (numpy.ndarray): array of x coordinates
        y (numpy.ndarray): array of y coordinates
        sensitivity (float): sensitivity of the monitoring. Lower numbers are
            high sensitivity. Value can't be zero.
        boundary (int): Number of consecutive observations identified as outliers
            to signal as disturbance
        q25 (numpy.ndarray): 25th percentile of residuals
        q75 (numpy.ndarray): 75th percentile of residuals
        detection_date (numpy.ndarray): 2D array signalling detection date of
            disturbances in days since 1970-01-01

    Args:
        mask (numpy.ndarray): A 2D numpy array containing pixels that should be
            monitored marked as ``1`` and pixels that should be excluded (marked
            as ``0``). Typically a stable forest mask when doing forest disturbance
            monitoring. If no mask is supplied all pixels are considered and
            a mask is created following the ``fit()`` call
        trend (bool): Indicate whether stable period fit is performed with
            trend or not
        harmonic_order (int): The harmonic order of the time-series regression
        sensitivity (float): sensitivity of the monitoring. Lower numbers are
            high sensitivity. Value can't be zero.
        boundary (int): Number of consecutive observations identified as outliers
            to signal as disturbance
        **kwargs: Used to set internal attributes when initializing with
            ``.from_netcdf()``
    """
    def __init__(self, trend=True, harmonic_order=3, sensitivity=1.5, mask=None,
                 boundary=3, **kwargs):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         boundary=boundary,
                         **kwargs)
        self.sensitivity = sensitivity
        self.q25 = kwargs.get('q25')
        self.q75 = kwargs.get('q75')
        self.monitoring_strategy = 'IQR'

    def fit(self, dataarray, method='OLS', **kwargs):
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)
        beta, residuals = self._fit(X, dataarray=dataarray, method=method,
                                    **kwargs)
        self.beta = beta
        q75, q25 = nan_percentile_axis0(residuals, np.array([75 ,25]))
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
