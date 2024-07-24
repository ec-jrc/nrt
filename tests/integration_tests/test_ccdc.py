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
import numpy as np

from nrt.monitor.ccdc import CCDC

# For now, because fixtures can't be parametrized and CCDC needs
# green_swir_history, this is done seperately. The package pytest-lazy-fixture
# could be used to also parametrize CCDC.


def test_fit_monitor(ndvi_history, green_swir_history, ndvi_monitoring_numpy,
                     forest_mask):
    green, swir = green_swir_history
    ccdc_monitor = CCDC(mask=forest_mask)
    ccdc_monitor.fit(dataarray=ndvi_history, green=green, swir=swir,
                     scaling_factor=10000)
    assert ccdc_monitor.beta.shape[0] == 6 # 2*2 harmonics + intercept + trend
    for array, date in zip(*ndvi_monitoring_numpy):
        ccdc_monitor.monitor(array=array, date=date)
    ccdc_monitor._report(layers=['mask', 'detection_date'],
                         dtype=np.int16)


def test_netcdf(ndvi_history, green_swir_history, tmp_path):
    nc_path = tmp_path / 'ccdc.nc'
    green, swir = green_swir_history
    ccdc_monitor = CCDC()
    ccdc_monitor.fit(dataarray=ndvi_history, green=green, swir=swir,
                     scaling_factor=10000)

    ccdc_monitor.to_netcdf(nc_path)
    ccdc_load = CCDC.from_netcdf(nc_path)
    assert ccdc_monitor == ccdc_load
