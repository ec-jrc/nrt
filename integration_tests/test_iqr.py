import pytest

import nrt
from nrt.iqr import Iqr


def test_fit_monitor(ndvi_history, ndvi_monitoring_numpy):
    iqr_monitor = Iqr(trend=False, harmonic_order=1)
    print(Iqr.harmonic_order)
    iqr_monitor.fit(dataarray=ndvi_history)
    assert iqr_monitor.beta.shape[0] == 3 # 2 harmonics + intercept
    for array, date in zip(*ndvi_monitoring_numpy):
        iqr_monitor.monitor(array=array, date=date)
    iqr_monitor._report()

