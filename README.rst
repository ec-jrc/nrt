***
nrt
***

*Python package for near real time detection of change in spatio-temporal datasets*

.. image:: https://badge.fury.io/py/nrt.svg
    :target: https://badge.fury.io/py/nrt

.. image:: https://readthedocs.org/projects/nrt/badge/?version=latest
    :target: https://nrt.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/ec-jrc/nrt/actions/workflows/build_and_test.yml/badge.svg
    :target: https://github.com/ec-jrc/nrt/actions/workflows/build_and_test.yml
    :alt: Build status


``nrt`` provides a standardized interface for Near Real Time monitoring of disturbances on satellite image time-series.
The package is optimized for fast computation and suitable for operational deployment at scale.
A typical operational use case of such package would be a system constantly receiving new satellite based acquisitions and generating alerts when an anomaly is detected.
Five monitoring frameworks from scientific literature on change detection are implemented and exposed via a common API.
All five monitoring framework share a common general approach which consists in modelling the "normal" behavior of the variable through time by fitting a linear model on a user defined stable history period and monitoring until a "break" is detected.
Monitoring starts right after the stable history period, and for each new incoming observation the observed value is compared to the predicted "normal" behavior.
When observations and predictions diverge, a "break" is detected.
A confirmed "break" typically requires several successive diverging observations, this sensitivity or rapid detection capacity depending on many variables such as the algorithm, its fitting and monitoring parameters, the noise level of the history period or the magnitude of the divergence. 
The five monitoring frameworks implemented are:

- Exponentially Weighted Moving Average (EWMA_) (Brooks et al., 2013) 
- Cumulative Sum of Residual (CuSum_) (Verbesselt et al., 2012; Zeileis et al., 2005). CuSum is one of the monitoring option of the ``bfastmonitor`` function available in the R package bfast_.
- Moving Sum of Residuals (MoSum_) (Verbesselt et al., 2012; Zeileis et al., 2005). MoSum is one of the monitoring option of the ``bfastmonitor`` function available in the R package bfast_.
- Continuous Change Detection and Classification of land cover (CCDC_, CMFDA_) (Zhu et al., 2012, 2014) - Partial implementation only of the original published method.
- InterQuantile Range (IQR) - Simple, unpublished outlier identification strategy described on stackexchange_.


Parts of this package are derived from Chris Holden's pybreakpoints_ and yatsm_ packages. Please see the copyright statements in the respective modules.

.. _EWMA: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6573358
.. _CMFDA: https://www.sciencedirect.com/science/article/pii/S0034425712000387
.. _CCDC: https://www.sciencedirect.com/science/article/pii/S0034425714000248#bbb0350
.. _CuSum: https://www.sciencedirect.com/science/article/pii/S0034425712001150
.. _MoSum: https://www.sciencedirect.com/science/article/pii/S0034425712001150
.. _stackexchange: https://stats.stackexchange.com/a/1153
.. _bfast: https://bfast.r-forge.r-project.org/
.. _pybreakpoints: https://github.com/ceholden/pybreakpoints
.. _yatsm: https://github.com/ceholden/yatsm



Documentation
=============

Learn more about nrt in its official documentation at https://nrt.readthedocs.io/en/latest/

  
Installation
============

.. code-block:: bash

    pip install nrt


Example usage
=============

The snippet below presents a near real time monitoring simulation. The input data is split in stable history and monitoring period; the monitoring class is instantiated (EWMA algorithm), a simple harmonic model is fitted on the history period, and new acquisition are passed to the monitor method one at the time. Note that in a real operational scenario where new observations come at a less frequent interval (e.g. every 5 or 8 days which coorespond to the revisit frequency of sentinel 2 and Landsat constellations respectively), the monitoring state can be saved on disk and reloaded when required.

.. code-block:: python

    import datetime

    from nrt.monitor.ewma import EWMA
    from nrt import data

    # Forest/non-forest mask
    mask = (data.romania_forest_cover_percentage() > 30).astype('int')

    # NDVI training and monitoring periods
    s2_cube = data.romania_20m()
    s2_cube['ndvi'] = (s2_cube.B8A - s2_cube.B4) / (s2_cube.B8A + s2_cube.B4)
    s2_cube = s2_cube.where(s2_cube.SCL.isin([4,5,7]))
    ndvi_history = s2_cube.ndvi.sel(time=slice('2015-01-01', '2018-12-31'))
    ndvi_monitoring = s2_cube.ndvi.sel(time=slice('2019-01-01', '2021-12-31'))

    # Instantiate monitoring class and fit stable history
    EwmaMonitor = EWMA(trend=False, mask=mask)
    EwmaMonitor.fit(dataarray=ndvi_history)

    # Monitor new observations
    for array, date in zip(ndvi_monitoring.values,
                           ndvi_monitoring.time.values.astype('M8[s]').astype(datetime.datetime)):
        EwmaMonitor.monitor(array=array, date=date)

    # At any time a monitoring report can be produced with EwmaMonitor.report(filename)
    # and state of the monitoring instance can be saved as netcdf with
    # EwmaMonitor.to_netcdf(filename)


About the authors
=================

Loïc Dutrieux works as a remote sensing researcher at the Joint Research Center (JRC) in Ispra, Italy. His work focuses on forest disturbances mapping and characterization from satellite image time-series.

Jonas Viehweger is a young researcher with a MSc in remote sensing from the university of Marburg, Germany. He developped a large part of the nrt package during his traineeship period at the Joint Research Center (JRC) in Ispra, Italy.

Chris Holden implemented many time-series change detection algorithms in python during his PhD at Boston university.


References
==========

Brooks, E.B., Wynne, R.H., Thomas, V.A., Blinn, C.E. and Coulston, J.W., 2013. On-the-fly massively multitemporal change detection using statistical quality control charts and Landsat data. IEEE Transactions on Geoscience and Remote Sensing, 52(6), pp.3316-3332.

Verbesselt, J., Zeileis, A. and Herold, M., 2012. Near real-time disturbance detection using satellite image time series. Remote Sensing of Environment, 123, pp.98-108.

Zeileis, A., Leisch, F., Kleiber, C. and Hornik, K., 2005. Monitoring structural change in dynamic econometric models. Journal of Applied Econometrics, 20(1), pp.99-121.

Zhu, Z., Woodcock, C.E. and Olofsson, P., 2012. Continuous monitoring of forest disturbance using all available Landsat imagery. Remote sensing of environment, 122, pp.75-91.

Zhu, Z. and Woodcock, C.E., 2014. Continuous change detection and classification of land cover using all available Landsat data. Remote sensing of Environment, 144, pp.152-171.


