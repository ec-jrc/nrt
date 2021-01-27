import numpy as np

import nrt.stability as cs


def test_ccdc_is_stable(stability_ccdc, threshold=3):
    residuals, slope, check_stability = stability_ccdc
    is_stable = cs.is_stable_ccdc(slope, residuals, threshold)
    np.testing.assert_array_equal(is_stable, check_stability)
