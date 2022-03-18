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
import xarray as xr

from nrt.monitor import BaseNrt


class CCDC(BaseNrt):
    """Monitoring using CCDC-like implementation

    Implementation loosely following method described in Zhu & Woodcock 2014.

    Zhu, Zhe, and Curtis E. Woodcock. 2014. “Continuous Change Detection and
    Classification of Land Cover Using All Available Landsat Data.” Remote
    Sensing of Environment 144 (March): 152–71.
    https://doi.org/10.1016/j.rse.2014.01.011.

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
        rmse (np.ndarray): 2D float array indicating RMSE for each pixel
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
    def __init__(self, trend=True, harmonic_order=2, sensitivity=3,
                 mask=None, boundary=3, **kwargs):
        super().__init__(mask=mask,
                         trend=trend,
                         harmonic_order=harmonic_order,
                         boundary=boundary,
                         **kwargs)
        self.sensitivity = sensitivity
        self.rmse = kwargs.get('rmse')
        self.monitoring_strategy = 'CCDC'

    def fit(self, dataarray, method='CCDC-stable', screen_outliers='CCDC_RIRLS',
            green=None, swir=None, scaling_factor=1, **kwargs):
        """Stable history model fitting

        If screen outliers is required, green and swir bands must be passed.

        The stability check will use the same sensitivity as is later used for
        detecting changes (default: 3*RMSE)

        Args:
            dataarray (xr.DataArray): xarray Dataarray including the historic
                data to be fitted
            method (string): Regression to use. See ``_fit()`` for info.
            screen_outliers (string): Outlier screening to use.
                See ``_fit()`` for info.
            green (xr.DataArray): Green reflectance values to be used by
                ``screen_outliers``.
            swir (xr.DataArray): Short wave infrared (SWIR) reflectance values
                to be used by ``screen_outliers``.
            scaling_factor (int): Optional Scaling factor to be applied to
                ``green`` and ``swir``.
            **kwargs: to be passed to ``_fit``
        """
        self.set_xy(dataarray)
        X = self.build_design_matrix(dataarray, trend=self.trend,
                                     harmonic_order=self.harmonic_order)
        self.beta, residuals = self._fit(X, dataarray,
                                         method=method,
                                         screen_outliers=screen_outliers,
                                         green=green, swir=swir,
                                         scaling_factor=scaling_factor,
                                         **kwargs)
        self.rmse = np.sqrt(np.nanmean(residuals ** 2, axis=0))

    def _update_process(self, residuals, is_valid):
        # TODO: Calculation is different for multivariate analysis
        # (mean of all bands has to be > sensitivity)
        with np.errstate(divide='ignore'):
            is_outlier = np.abs(residuals) / self.rmse > self.sensitivity
        # Update process
        if self.process is None:
            self.process = np.zeros_like(residuals, dtype=np.uint8)
        self.process = np.where(is_valid,
                                self.process * is_outlier + is_outlier,
                                self.process)
