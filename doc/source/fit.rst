.. _fitting:

Fitting & Outlier Screening
***************************

Fitting is achieved by calling ``.fit()`` on an instantiated monitoring class.

In general the default arguments for each monitoring class correspond to the fitting
which was used in the corresponding publication. However since the
classes are not bound to the fit that was used in the publication it is entirely possible
to use any combination of fitting arguments with any monitoring class.

Fitting works by passing an ``xarray.DataArray`` and specifying a fitting method.
Optionally a method to screen outliers in the time-series can also be passed
to the fit call.

Screen outliers
===============

Outlier screening happens before the fitting and is designed to remove unwanted outliers
in the time-series. When using optical satellite data those outliers are mostly unwanted
clouds, cloud shadows and snow.

Shewhart
^^^^^^^^

.. code-block:: python
    
    nrt_class.fit(dataarray, screen_outliers='Shewhart', L=5)
    
This outlier screening is using Shewhart control charts to remove outliers.
The optional parameter ``L`` defines how sensitive the outlier screening is.

With this method, first an OLS fit is carried out using the arguments passed during
instantiation. Then the standard deviation :math:`\sigma` of residuals is computed and all observations with
residuals larger than :math:`L\cdot\sigma` are screened out.


CCDC-RIRLS
^^^^^^^^^^

While Shewhart outlier screening could work for any type of time series, the default outlier screening
used by CCDC is tailored for optical satellite time series to mask out clouds and
cloud shadows.

.. code-block:: python
    
    nrt_class.fit(dataarray, screen_outliers='CCDC-RIRLS', 
                  green=xr_green, swir=xr_swir, scaling_factor=10000)
    
This screening uses hard-coded thresholds of the short-wave infrared (SWIR) and green bands
to detect clouds and cloud shadows. For this, reflectance values of the green and 
SWIR bands need to be passed as ``xarray.DataArrays``. Originally the bands 2 (0.52-0.60 µm) and 5 (1.55-1.75 µm) 
of the Landsat 5 Thematic Mapper were used.

If other sensors like Sentinel 2 are used, which supply data with a scaling factor, the optional parameter
``scaling_factor`` needs to be set appropriately to bring the values to a 0-1 range.

To screen out clouds, CCDC-RIRLS uses a Robust Iteratively Reweighted Least Squares fit to reduce the influence
of outliers on the fit. See the chapter about RIRLS for more details.

Do note that the RIRLS fit is quite computationally intensive.


Fitting
=======

In general when trying to fit a temporal signature it is advisable to fit it on a stable part
of the time-series which doesn't include structural changes.
For this there are two fitting methods (ROC and CCDC-stable) available that aim to achieve a fit on a stable
part of the time-series.
The other two fitting methods (OLS, RIRLS) always fit a model on the entire history period, so if
a lot of disturbances happened during the history period, the fitting results with these
two methods might deliver worse results. Especially OLS however is much less computationally expensive than
ROC and CCDC-stable.

OLS
^^^^

.. code-block:: python
    
    nrt_class.fit(dataarray, method='OLS')
    
This carries out an ordinary least squares fit. All other available fitting methods in this package
are at some point based on this fit.

RIRLS
^^^^^^

.. code-block:: python
    
    nrt_class.fit(dataarray, method='RIRLS', maxiter=50)

The Robust Iteratively Reweighted Least Squares fit isn't the default for any nrt monitoring class, it's
main purpose is in the outlier screening method CCDC-RIRLS.

By iteratively reweighting each observation in the time-series, a fit is reached which is less influenced by
outliers in the time-series.

This process can take a lot of iterations and thus can become very computationally expensive. The maximum number
of iterations can be controlled by setting ``maxiter``. There are also many more possible parameters to modify.
For a complete list see the api documentation for ``RIRLS``.

ROC
^^^^

.. code-block:: python
    
    nrt_class.fit(dataarray, method='ROC', alpha=0.05)

Reverse Ordered Cumulative Sums (ROC) works by applying the same type of monitoring logic as in CuSum to the fitting.
In particular this means, that the fitting period is gradually increased backwards in time starting from the
end of the entire history period (so in reverse order). The period is increased as long as the
cumulative sum of residuals is within a certain threshold which depends on ``alpha``.

As soon as the threshold is crossed, it is likely that there was a structural break in the history period and thus
the rest of the time series before the threshold was crossed will not be used for fitting the model.

``alpha`` is the significance of the detected structural break. So the lower ``alpha`` the lower the sensitivity
for breaks in the time-series.


CCDC-stable
^^^^^^^^^^^^

.. code-block:: python
    
    nrt_class.fit(dataarray, method='CCDC-stable', threshold=3)

With CCDC-stable, models are first fit using an OLS regression. 
Those models are then checked for stability.

Stability is given if:

1.             slope / RMSE < threshold
2. first observation / RMSE < threshold
3.  last observation / RMSE < threshold


Since the slope of the model is one of the test conditions, it is required for ``trend`` to be ``True``
during instantiation of the monitoring class.

If a model is not stable, the two oldest
acquisitions are removed, a model is fit using this shorter
time-series and again checked for stability. This process continues until the model is stable
or until not enough observations are left, at which point the time-series will get marked as
unstable and not be fit.

.. note::
    This process is slightly different to the one described in Zhu & Woodcock 2014,
    since with the nrt package no new observations can be added during fitting.