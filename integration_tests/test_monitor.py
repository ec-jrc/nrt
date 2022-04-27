# Copyright (C) 2022 European Union (Joint Research Centre)
#
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#   https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.

from pathlib import Path
import pytest
import numpy as np

from nrt.monitor import iqr, ewma, cusum, mosum

monitor_params = {
    'EWMA': pytest.param(ewma.EWMA, {'trend': False, 'L': 5}, 5,
                         marks=pytest.mark.ewma),
    'IQR': pytest.param(iqr.IQR, {'trend': False, 'harmonic_order': 1}, 3,
                        marks=pytest.mark.iqr),
    'CUSUM': pytest.param(cusum.CuSum, {}, 6,
                          marks=pytest.mark.cusum),
    'MOSUM': pytest.param(mosum.MoSum, {}, 6,
                          marks=pytest.mark.mosum)
}

@pytest.mark.parametrize('monitor_cls, kwargs, beta', monitor_params.values(),
                         ids=monitor_params.keys())
def test_fit_monitor(monitor_cls, kwargs, beta,
                     ndvi_history, ndvi_monitoring_numpy, forest_mask):
    monitor_ = monitor_cls(**kwargs, mask=forest_mask)
    monitor_.fit(dataarray=ndvi_history, **kwargs)
    assert monitor_.beta.shape[0] == beta
    for array, date in zip(*ndvi_monitoring_numpy):
        monitor_.monitor(array=array, date=date)
    monitor_._report(layers=['mask', 'detection_date'],
                     dtype=np.int16)


@pytest.mark.parametrize('monitor_cls, kwargs, beta', monitor_params.values(),
                         ids=monitor_params.keys())
def test_netcdf(monitor_cls, kwargs, beta, ndvi_history, tmp_path):
    nc_path = tmp_path / 'monitor.nc'
    monitor_ = monitor_cls(**kwargs)
    monitor_.fit(dataarray=ndvi_history, **kwargs)

    monitor_.to_netcdf(nc_path)
    monitor_load = monitor_cls().from_netcdf(nc_path)
    assert monitor_ == monitor_load

