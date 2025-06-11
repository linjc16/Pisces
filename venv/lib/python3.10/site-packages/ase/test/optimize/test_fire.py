import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import FIRE


@pytest.mark.optimize()
@pytest.mark.slow()
def test_fire():
    a = bulk('Au')
    a *= (2, 2, 2)

    a[0].x += 0.5

    a.calc = EMT()

    opt = FIRE(a, dtmax=1.0, dt=1.0, maxstep=100.0, downhill_check=False)
    opt.run(fmax=0.001)
    e1 = a.get_potential_energy()
    n1 = opt.nsteps

    a = bulk('Au')
    a *= (2, 2, 2)

    a[0].x += 0.5

    a.calc = EMT()

    reset_history = []

    def callback(a, r, e, e_last):
        reset_history.append([e - e_last])

    opt = FIRE(a, dtmax=1.0, dt=1.0, maxstep=100.0, downhill_check=True,
               position_reset_callback=callback)
    opt.run(fmax=0.001)
    e2 = a.get_potential_energy()
    n2 = opt.nsteps

    assert abs(e1 - e2) < 1e-6
    assert n2 < n1
    assert all(np.array(reset_history) > 0)
