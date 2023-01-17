"""
This tutorial illustrates the emulation of a near real time monitoring scenario on synthetic data
by all available monitoring methods. The experiment is concluded by an accuracy assessment.
The different sections of the tutorial are:
    - Simulation of synthetic data
    - Harmonic regression parameters fitting for a model
    - MOnitoring
    - Accuracy assessment
    - Comparison of all monitoring methods implemented
    - COnclusion:
        - ONly default parameters tested
        - Synthetic data useful for ensuring that everything works
        - Rough approximation of the actual temporal behaviour of vegetation (see timesync for a more elaborated phenology fit)
        - Having a standardized interface makes intercomparison really easy (all monitoring methods are called via the same API)
"""
import datetime

import numpy as np
import xarray as xr
from nrt import data
from nrt.monitor.ewma import EWMA
import matplotlib.pyplot as plt


# The data module of the nrt package contains functionalities to create synthetic data with controlled parameters such as position of structural change, phenological amplitude, noise level, etc
# One such example can be visualized using the ``make_ts`` function, which creates a single time-series.

dates = np.arange('2018-01-01', '2022-06-15', dtype='datetime64[W]')
break_idx = 50
ts = data.make_ts(dates=dates, break_idx=break_idx)

plt.plot(dates, ts)
plt.axvline(x=dates[break_idx], color='magenta')
# plt.show()

# The spatial counterpart of ``make_ts`` is ``make_cube`` and its main argument is a ``Dataset`` of simulation parameters
# that can be generated with the ``make_cube_parameters`` function.

params_ds = data.make_cube_parameters(shape=(50,50),
                                      n_outliers_interval=(0,5),
                                      n_nan_interval=(0,7),
                                      break_idx_interval=(105,dates.size - 20))
print('Early breakpoint: %s' % dates[105])
print('Late breakpoint: %s' % dates[dates.size - 20])
cube = data.make_cube(dates=dates, params_ds=params_ds)
print(cube)

# In the ndvi datacube created, 50 percents of the pixels contain a breakpoint occuring between 2020-01-02 and 2022-01-20
# The ``break_idx`` variable of the ``params_ds`` ``Dataset`` informs on the presence or absence of a breakpoint, and its position.

# For the monitoring emulation, we'll consider all the pixels of the datacube (no mask) and define the 2018-01-01 to 2019-12-31 period as the stable history period and all subsequent dates as monitoring. We know from the time-series simulation parameters that the stable history period is indeed free of breakpoints.
# In a real life near real time monitoring use case, fitting and monitoring are occuring separately; we therefore need to split the datacube created in two.

cube_history = cube.sel(time=slice('2018-01-01','2019-12-31'))
cube_monitor = cube.sel(time=slice('2020-01-01', '2022-12-31'))


# INstantiation and fitting
# After fitting the harmonic fit parameters for each pixel individually is stored in the instance of the EWMA class
# Note that in a real life scenario, several days may pass between fitting and the next observation or between consecutive observations. The fit parameters or ongoing monitoring vasriables are then usually stored to disk in a NetCDF file. See the ``to_netcdf()`` method 

EwmaMonitor = EWMA(trend=False)
EwmaMonitor.fit(dataarray=cube_history)

# Emulate monitoring
# During monitoring each new observation need to be passed to the monitor method method as a numpy array. Since we currently have these observation in a xarray DataArray structure, we need to unpack each temporal slice as an (array, date) couple
for array, date in zip(cube_monitor.values,
                       cube_monitor.time.values.astype('M8[s]').astype(datetime.datetime)):
    EwmaMonitor.monitor(array=array, date=date)


# To retrieve the result of the monitoring we can simply run the ``_report`` method.
print(EwmaMonitor._report(layers=['mask', 'detection_date'], dtype=np.uint16))


