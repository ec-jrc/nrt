r"""
Synthetic disturbance data
==========================
This example illustrates the simulation of a near real time monitoring scenario on synthetic data.
The EWMA approach instantiated from ``nrt.monitor.ewma import EWMA`` is used for monitoring and detection
of the artificially generated breakpoints and the experiment is concluded by a simple accuracy assessment.
"""

#############################################################
# Synthetic data generation
# =========================
#
# The data module of the nrt package contains functionalities to create synthetic
# data with controlled parameters such as position of structural change, phenological
# amplitude, noise level, etc
# One such example can be visualized using the ``make_ts`` function, which
# creates a single time-series.
import random

import numpy as np
from nrt import data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

dates = np.arange('2018-01-01', '2022-06-15', dtype='datetime64[W]')
fig, axes = plt.subplots(3,3, constrained_layout=True)
for row, amplitude in zip(axes, [0.1, 0.2, 0.3]):
    for ax, noise in zip(row, [0.02, 0.05, 0.1]):
        break_idx = random.randint(30,100)
        ts = data.make_ts(dates=dates,
                          break_idx=break_idx,
                          sigma_noise=noise,
                          amplitude=amplitude)
        ax.plot(dates, ts)
        ax.axvline(x=dates[break_idx], color='magenta')
        ax.set_ylim(-0.1,1.1)
        ax.set_title('Amplitude: %.1f\nsigma noise: %.2f' % (amplitude, noise),
                     fontsize=11)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("\n%Y"))
        ax.tick_params( axis='x', which='both', bottom=False, top=False,
                       labelbottom=False)
fig.supxlabel('Time')
fig.supylabel('NDVI')
plt.show()

#################################################################
# The spatial counterpart of ``make_ts`` is ``make_cube`` and its main argument
# is an ``xarray.Dataset`` of simulation parameters that can be generated
# with the ``make_cube_parameters`` function.
# The data cube generated is a standard univariate ``xarray.DataArray`` with
# ``x``, ``y`` and ``time`` dimensions. Each pixel in the spatial dimensions contains
# a time-series of simulated values with varying levels of noise, seasonality, outliers
# and in some cases a structural break point

params_ds = data.make_cube_parameters(shape=(50,50),
                                      n_outliers_interval=(0,5),
                                      n_nan_interval=(0,7),
                                      break_idx_interval=(105,dates.size - 20))
# Convert break_idx to dates
print('Early breakpoint: %s' % dates[105])
print('Late breakpoint: %s' % dates[dates.size - 20])
cube = data.make_cube(dates=dates, params_ds=params_ds)

#################################################################
# In the ndvi datacube created, 50 percents of the pixels contain a break point
# occuring between 2020-01-02 and 2022-01-20. The ``break_idx`` variable of
# the ``params_ds`` ``Dataset`` informs on the presence or absence of a break point,
# and its position.

###################################################################
# Simulation of an NRT monitoring scenario
# ==================================================
# For the simulating a near real time monitoring scenario, we consider all the pixels
# of the datacube (no mask) and define the 2018-01-01 to 2019-12-31 period as the
# stable history period and all subsequent dates as monitoring. We know from the
# time-series simulation parameters that the stable history period is indeed free of breakpoints.
# In a real life near real time monitoring use case, fitting and monitoring are
# occuring separately; we therefore need to split the datacube created in two.
#
# After that instantiation of the ``EWMA`` class and stable history takes place.
# The harmonic fit parameters for each pixel is stored in the instance
# of the ``EWMA`` class
# Note that in a real life scenario, several days may pass between fitting and the
# next observation, or between consecutive observations. The fit parameters or
# ongoing monitoring variables are then usually stored to disk in a NetCDF file.
# See the ``to_netcdf()`` method for more details. 
# During monitoring each new observation needs to be passed to the monitor method 
# as a numpy array. Since we currently have these observations in an xarray DataArray
# structure, we need to unpack each temporal slice as an (array, date) tuple

import datetime

from nrt.monitor.ewma import EWMA

cube_history = cube.sel(time=slice('2018-01-01','2019-12-31'))
cube_monitor = cube.sel(time=slice('2020-01-01', '2022-12-31'))

# Monitoring class instantiation and fitting
monitor = EWMA(trend=False, harmonic_order=1, lambda_=0.3, sensitivity=4,
               threshold_outlier=10)
monitor.fit(dataarray=cube_history)

# Monitor every date of the ``cube_monitor`` DataArray
for array, date in zip(cube_monitor.values,
                       cube_monitor.time.values.astype('M8[s]').astype(datetime.datetime)):
    monitor.monitor(array=array, date=date)


############################################################################
# Monitoring performances evaluation
# ==================================
# Assessing the performance of a time-series monitoring algorithm can be a complex
# task that depends on the specific use case and what the user wants to emphasize.
# A user valuing rapid detection will chose an assessment approach that puts extra
# weight on the temporal aspect or penalize late detections, while if timeliness
# is not a requirement, accuracy assessment will resemble standard spatial validation.
# In the present example we work with a temporal threshold for which 6 months is the
# default value. This approach to accuracy assessment implies that any breakpoint
# occuring outside of the 6 months periods after the simulated breakpoint (considered ground thruth)
# is considered comission error. Absence of detection during that same period would then be
# an omission, detections during the period are true positives, and absence of detection
# on stable time-series are true negatives.
# Note that alternative accuracy assessment approaches exist; see for instance [1]_ who
# chose to use ``PixelYears`` as their sampling units, or [2]_ who introduced the
# concept of a time weighted F1 score, hence considerating simultaneously detection
# speed and spatial accuracy in a single index.

def accuracy(nrtInstance, params_ds, dates, delta=180):
    """Compute accuracy metrics (precision, recall) of a nrt simulation on synthetic data

    Args:
        nrtInstance: Instance of a nrt monitoring class used for monitoring
        params_ds: Time-series generation paramaters
        dates: Array of numpy.datetime64 dates used for synthetic time-series generation
        delta (int): Time delta in day after a reference break for a detected break
            to be considered True Positive.
    """
    detection_date = nrtInstance._report(layers=['detection_date'], dtype=np.uint16)
    dates_true = np.where(params_ds.break_idx != -1,
                          dates[params_ds.break_idx.values],
                          np.datetime64('NaT'))
    dates_true_bound = dates_true + np.timedelta64(delta)
    dates_pred = np.datetime64('1970-01-01') + np.timedelta64(1) * detection_date
    dates_pred[dates_pred == np.datetime64('1970-01-01')] = np.datetime64('NaT')
    # Computes arrays of TP, FP, FN (they should be mutually exclusive)
    TP = np.where(np.logical_and(dates_pred >= dates_true, dates_pred <= dates_true_bound),
                  1, 0)
    FP = np.where(np.logical_and(TP == 0, ~np.isnat(dates_pred)), 1, 0)
    FN = np.where(np.logical_and(np.isnat(dates_pred), ~np.isnat(dates_true)), 1, 0)
    precision = TP.sum() / (TP.sum() + FP.sum())
    recall = TP.sum() / (TP.sum() + FN.sum())
    return precision, recall

print(accuracy(monitor, params_ds, dates))

####################################################################
# White noise sensitivity analysis
# ================================
# To go one step further we can assess and visualize how these accuracy measures
# vary with the amount of noise present in the synthetic data.
# For that we define a new function encompassing all the steps of data generation,
# instantiation, fitting and monitoring
#
# The increase in recall at low noise levels is probably due to the extreme outliers
# filtering feature of the EWMA monitoring process, OUtliers that exceed ``threshold_outlier``
# times the standard deviation of the fit residuals are considered extreme
# outliers (often clouds or artifacts) in real images, and do not contribute to the monitoring
# process. With such low noise levels, that threshold is easily reached and breaks missed.

def make_cube_fit_and_monitor(dates, noise_level):
    params_ds = data.make_cube_parameters(shape=(20,20),
                                          n_outliers_interval=(4,5),
                                          n_nan_interval=(3,4),
                                          sigma_noise_interval=(noise_level, noise_level),
                                          break_idx_interval=(105,dates.size - 20))
    cube = data.make_cube(dates=dates, params_ds=params_ds)
    cube_history = cube.sel(time=slice('2018-01-01','2019-12-31'))
    cube_monitor = cube.sel(time=slice('2020-01-01', '2022-12-31'))
    # Monitoring class instantiation and fitting
    monitor = EWMA(trend=False, harmonic_order=1, lambda_=0.3, sensitivity=4,
                   threshold_outlier=10)
    monitor.fit(dataarray=cube_history)
    # Monitor every date of the ``cube_monitor`` DataArray
    for array, date in zip(cube_monitor.values,
                           cube_monitor.time.values.astype('M8[s]').astype(datetime.datetime)):
        monitor.monitor(array=array, date=date)
    return params_ds, monitor

noises = [0.02, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.2]
prs = []
for noise in noises:
    params_ds, monitor = make_cube_fit_and_monitor(dates, noise)
    prs.append(accuracy(monitor, params_ds, dates))

precisions, recalls = zip(*prs)
plt.plot(noises, precisions, label='Precision')
plt.plot(noises, recalls, label='Recall')
plt.xlabel('Noise level')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

###########################################################################
# References
# ----------
#
# .. [1] Bullock, E.L., Woodcock, C.E. and Holden, C.E., 2020. Improved
#        change monitoring using an ensemble of time series algorithms.
#        Remote Sensing of Environment, 238, p.111165.
#
# .. [2] Viehweger, J., 2021. Comparative Assessment of Near Real-Time Forest
#        Disturbance Detection Algorithms. Master thesis, Philipps Universitat
#        Marburg
