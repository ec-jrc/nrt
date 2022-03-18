***
nrt
***

*Python package for near real time detection of change in spatio-temporal datasets*

Parts of this package are derived from Chris Holden's pybreakpoints and yatsm packages. Please see the copyright statements in the respective modules.

Monitoring frameworks implemented
=================================

- EWMA (Brooks et al., 2013)
- CUSUM (Verbesselt et al., 2012)
- MOSUM (Verbesselt et al., 2012)
- CCDC (Zhu et al., 2014)
- IQR
  
Installation
============

.. code-block:: bash

    pip install nrt
    # Or alternatively
    pip install git+https://github.com/ec-jrc/nrt.git


Example usage
=============

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
