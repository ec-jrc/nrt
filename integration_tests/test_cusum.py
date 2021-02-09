from nrt.monitor.cusum import CuSum


def test_fit_monitor(ndvi_history, ndvi_monitoring_numpy):
    cusum_monitor = CuSum()
    cusum_monitor.fit(dataarray=ndvi_history)
    assert cusum_monitor.beta.shape[0] == 6  # 2*2 harmonics + intercept + trend
    for array, date in zip(*ndvi_monitoring_numpy):
        cusum_monitor.monitor(array=array, date=date)
    cusum_monitor._report()
