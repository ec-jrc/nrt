from nrt.CCDC import CCDC


def test_fit_monitor(ndvi_history, green_swir_history, ndvi_monitoring_numpy):
    green, swir = green_swir_history
    brooks_monitor = CCDC()
    brooks_monitor.fit(dataarray=ndvi_history, green=green, swir=swir,
                       scaling_factor=10000)
    assert brooks_monitor.beta.shape[0] == 6 # 2*2 harmonics + intercept + trend
    for array, date in zip(*ndvi_monitoring_numpy):
        brooks_monitor.monitor(array=array, date=date)
    brooks_monitor._report()