from nrt.monitor.cusum import CuSum


def test_fit_monitor(ndvi_history, green_swir_history, ndvi_monitoring_numpy):
    green, swir = green_swir_history
    cusum_monitor = CuSum()
    cusum_monitor.fit(dataarray=ndvi_history, green=green, swir=swir,
                     scaling_factor=10000)
    assert cusum_monitor.beta.shape[0] == 6  # 2*2 harmonics + intercept + trend
    # for array, date in zip(*ndvi_monitoring_numpy):
    #     cumsum_monitor.monitor(array=array, date=date)
    # cumsum_monitor._report()
