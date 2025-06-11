import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.fd import calculate_numerical_stress


def test_emt_stress():

    a = bulk('Cu', 'fcc')
    a.calc = EMT()
    a.set_cell(np.dot(a.cell,
                      [[1.02, 0, 0.03],
                       [0, 0.99, -0.02],
                       [0.1, -0.01, 1.03]]),
               scale_atoms=True)
    a *= (1, 2, 3)
    a.rattle()
    # Verify analytical stress tensor against numerical value
    s_analytical = a.get_stress()
    s_numerical = calculate_numerical_stress(a, 1e-5)
    s_p_err = 100 * (s_numerical - s_analytical) / s_numerical
    print('Analytical stress:\n', s_analytical)
    print('Numerical stress:\n', s_numerical)
    print('Percent error in stress:\n', s_p_err)
    assert np.all(abs(s_p_err) < 1e-5)
