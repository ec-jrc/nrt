"""Removing outliers

Functions defined in this module always use a 2D array containing the dependant
variables (y) and return y with outliers set to np.nan.
These functions are meant to be called in ``nrt.BaseNrt._fit()``

Citations:

- Brooks, E.B., Wynne, R.H., Thomas, V.A., Blinn, C.E. and Coulston, J.W., 2013.
  On-the-fly massively multitemporal change detection using statistical quality
  control charts and Landsat data. IEEE Transactions on Geoscience and Remote Sensing,
  52(6), pp.3316-3332.

- Zhu, Zhe, and Curtis E. Woodcock. 2014. “Continuous Change Detection and
  Classification of Land Cover Using All Available Landsat Data.” Remote
  Sensing of Environment 144 (March): 152–71.
  https://doi.org/10.1016/j.rse.2014.01.011.
"""
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

from nrt.fit_methods import rirls, ols
from nrt.log import logger


def shewhart(X, y, L=5, **kwargs):
    """Remove outliers using a Shewhart control chart

    As described in Brooks et al. 2014, following an initial OLS fit, outliers are
    identified using a shewhart control chart and removed.

    Args:
        X ((M, N) np.ndarray): Matrix of independant variables
        y ({(M,), (M, K)} np.ndarray): Matrix of dependant variables
        L (float): control limit used for outlier filtering. Must be a positive
            float. Lower values indicate stricter filtering. Residuals larger
            than L*sigma will get screened out
        **kwargs: not used

    Returns:
        y(np.ndarray): Dependant variables with outliers set to np.nan
    """
    beta_full, residuals_full = ols(X, y)
    # Shewhart chart to get rid of outliers (clouds etc)
    sigma = np.nanstd(residuals_full, axis=0)
    shewhart_mask = np.abs(residuals_full) > L * sigma
    y[shewhart_mask] = np.nan
    return y


def ccdc_rirls(X, y, green, swir, scaling_factor=1, **kwargs):
    """Screen for missed clouds and other outliers using green and SWIR band

    Args:
        X ((M, N) np.ndarray): Matrix of independant variables
        y ((M, K) np.ndarray): Matrix of dependant variables
        green (np.ndarray): 2D array containing spectral values
        swir (np.ndarray): 2D array containing spectral values (~1.55-1.75um)
        scaling_factor (int): Scaling factor to bring green and swir values
            to reflectance values between 0 and 1

    Returns:
        np.ndarray: y with outliers set to np.nan
    """
    # 1. estimate time series model using rirls for green and swir
    # TODO could be sped up, since masking is the same for green and swir
    g_beta, g_residuals = rirls(X, green, **kwargs)
    s_beta, s_residuals = rirls(X, swir, **kwargs)
    # Update mask using thresholds
    is_outlier = np.logical_or(g_residuals > 0.04*scaling_factor,
                               s_residuals < -0.04*scaling_factor)

    removed = np.count_nonzero(is_outlier) / np.count_nonzero(~np.isnan(green))
    if removed > 0.5:
        logger.warn('More than 50% of pixels have been removed as outliers. '
                    'Check if scaling_factor has been set correctly.')
    logger.debug('%.2f%% of (non nan) pixels removed.',
                 removed * 100)

    y[is_outlier] = np.nan
    return y
