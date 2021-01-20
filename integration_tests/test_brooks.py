import pytest

import nrt
from nrt.brooks import Brooks


def test_fit_monitor(ndvi_history_dataarray, ndvi_monitoring_numpy):
    brooks_monitor = Brooks(trend=False)
    brooks_monitor.fit(dataarray=ndvi_history_dataarray, L=5)
    assert brooks_monitor.beta.shape[0] == 5 # 2*2 harmonics + intercept
    for array, date in zip(*ndvi_monitoring_numpy):
        brooks_monitor.monitor(array=array, date=date)
    brooks_monitor._report()

