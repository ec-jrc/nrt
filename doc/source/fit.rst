Fit
***

Fitting is achieved by calling ``.fit()`` on an instantiated monitoring class.

In general the default arguments for each monitoring class correspond to the fitting
which was used in the corresponding publication. However since the
classes are not bound to the fit that was used in the publication it is entirely possible
to use any combination of fitting arguments with any monitoring class.

Fitting works by passing an xarray data array and specifying a fitting method.
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
instantiation. Then the standard deviation .. math::`\sigma` of residuals is computed and all observations with
residuals larger than .. math::`L\cdot\sigma` are screened out.


CCDC-RIRLS
^^^^^^^^^^

While Shewhart outlier screening could work for any type of time series, the outlier screening
used by default by CCDC is tailored for optical satellite time series to mask out clouds and
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


OLS
^^^^

RIRLS
^^^^^^

ROC
^^^^

CCDC-stable
^^^^^^^^^^^^

