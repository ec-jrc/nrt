from nrt.monitor.mosum import MoSum


def test_fit_monitor(ndvi_history, ndvi_monitoring_numpy):
    mosum_monitor = MoSum()
    mosum_monitor.fit(dataarray=ndvi_history)
    assert mosum_monitor.beta.shape[0] == 6  # 2*2 harmonics + intercept + trend
    for array, date in zip(*ndvi_monitoring_numpy):
        mosum_monitor.monitor(array=array, date=date)
    mosum_monitor._report()
