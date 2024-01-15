r"""
Parallel model fitting
======================
The most computationally expensive part of a typical nrt workflow is the fitting
of a harmonic model over the stable history period. Starting with version ``0.2.0``,
``nrt`` uses multithreading to further speed-up the already fast model fitting.
This example illustrates how multithreading can be enabled and adjusted to your use case.
"""

##############################################################
# Confirgure multithreading options of linear algebra library
# ===========================================================
#
# Most of the low level computation/numerical optimization occuring during model
# fitting with nrt relies on a linear algebra library. These libraries often implement
# low level methods with built-in multi-threading. ``nrt`` implements multi-threading
# thanks to ``numba`` on a different, higher level.
# To prevent nested parallelism that would result in over-subscription and potentially
# reduce performances, it is recommanded to disable the built in multi-threading
# of the linear algebra library being used.
# Depending on how ``numpy`` was installed, it will rely on one of the three linear
# algebra libraries which are OpenBLAS, MKL or BLIS. At the time of writing this
# tutorial, pipy wheels (obtain when installing ``numpy`` using pip) are shipped
# with OpenBLAS, while a conda installation from the default channel will come with
# MKL. All three libraries use an environmental variable to control threading 
# (``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` and ``BLIS_NUM_THREADS``); in the
# present example, we set them all to ``'1'`` directly from within python.
# Although knowing which library is used on your system would allow you to remove
# the unnecessary configuration lines, it is not entirely necessary.
import os
# Note that 1 is a string, not an integer
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['BLIS_NUM_THREADS'] = '1'

##############################################################
# Create benchmark data
# =====================
#
# Using the synthetic data generation functionalities of the package, we can create
# an xarray DataArray for benchmark. Note that in order to keep the compilation time
# of this tutorial manageable we limit the size of that object to 200 by 200 pixels.
# While this is significantly smaller than e.g. a Sentinel2 MGRS tile, it is sufficient
# to illustrate differences in fitting time among various fitting strategies
import xarray as xr
import numpy as np
from nrt import data

# Create synthetic ndvi data cube
dates = np.arange('2018-01-01', '2020-12-31', dtype='datetime64[W]')
params_ds = data.make_cube_parameters(shape=(200,200), unstable_proportion=0)
cube = data.make_cube(dates=dates, params_ds=params_ds)
# We also create a very small cube for running each fitting method once before
# the benchmark, ensuring compilation of the jitted functions and fair comparison
cube_sub = cube.isel(indexers={'x': slice(1,5), 'y': slice(1,5)})


##############################################################
# Benchmark fitting time of all methods 
# =====================================
# 
# Note that we are only interested in fitting time and therefore use a single
# class instance for the benchmark. The time required for any subsequent .monitor()
# call is usually negligible and as a consequence not included in this benchmark.
# We use here ``CuSum`` but any of the monitoring classes could be used and
# would produce the same results.
import time
import itertools
from collections import defaultdict
from nrt.monitor.cusum import CuSum
import matplotlib.pyplot as plt

# Benchmark parameters
benchmark_dict = defaultdict(dict)
monitor = CuSum()
methods = ['OLS', 'RIRLS', 'CCDC-stable', 'ROC']
threads = range(1,4)

# Make sure all numba jitted function are compiled
monitor_ = CuSum()
[monitor_.fit(cube_sub, method=method) for method in methods]

# Benchmark loop
for method, n_threads in itertools.product(methods, threads):
    t0 = time.time()
    monitor.fit(cube, n_threads=n_threads, method=method)
    t1 = time.time()
    benchmark_dict[method][n_threads] = t1 - t0

# Visualize the results
index = np.arange(len(methods))
for idx, n in enumerate(threads):
    values = [benchmark_dict[method][n] for method in methods]
    plt.bar(index + idx * 0.2, values, 0.2, label='%d thread(s)' % n)

plt.xlabel('Fitting method')
plt.ylabel('Time (seconds)')
plt.title('Fitting time')
plt.xticks(index + 0.2, methods)
plt.legend()
plt.tight_layout()
plt.show()

##############################################################
# From the results above we notice large differences in fitting time among fitting
# methods. Unsurprisingly, OLS is the fastest, which is expected given that all
# other methods use OLS complemented with some additional, sometimes iterative
# refitting, etc... All methods but ``ROC`` for which parallel fitting hasn't been
# implemented, benefit from using multiple threads.
# Note that a multithreading benefit can only be observed as long as the number
# threads is lower than the computing resources available. The machine used for
# compiling this tutorial is not meant for heavy computation and obviously has limited 
# resources as shown by the cpu_count below
import multiprocessing
print(multiprocessing.cpu_count())


##############################################################
# Further considerations
# ======================
#
# A deployment at scale may involve several levels of parallelization. The multi-threaded
# example illustrated above is made possible thanks to the numba parallel accelerator.
# However, it is also very common to handle the earlier steps of data loading and
# data pre-processing with ``dask.distributed``, which facilitates lazy and distributed
# computation. There is no direct integration between the two parallelism mechanisms
# and while calling ``.fit()`` on a lazy distributed dask array is possible, the lazy
# evaluation cannot be preserved and all the input data need to be evaluated and
# loaded in memory
from nrt import data

# Lazy load test data using dask
cube = data.romania_10m(chunks={'x': 20, 'y': 20})
vi_cube = (cube.B8 - cube.B4) / (cube.B8 + cube.B4)
print(vi_cube)
monitor = CuSum()
monitor.fit(vi_cube, method='OLS', n_threads=3)
print(type(monitor.beta))
