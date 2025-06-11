import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import FIRE2


@pytest.mark.optimize
@pytest.mark.slow
def test_fire2():
    def system_setup():
        a = bulk('Au')
        a *= (2, 2, 2)
        a[0].x += 0.5
        a.calc = EMT()
        return a

    a = system_setup()

    fire_parameters = {'dt': 0.1,
                       'maxstep': 0.2,
                       'dtmax': 1.0,
                       'Nmin': 20,
                       'finc': 1.1,
                       'fdec': 0.5,
                       'astart': 0.25,
                       'fa': 0.99}

    opt = FIRE2(a,
                **fire_parameters)
    opt.run(fmax=0.001)
    e1 = a.get_potential_energy()
    n1 = opt.nsteps

    a = system_setup()

    reset_history = []

    def callback(a, r, e, e_last):
        reset_history.append([e - e_last])

    opt = FIRE2(a,
                use_abc=True,
                position_reset_callback=callback,
                **fire_parameters)
    opt.run(fmax=0.001)
    e2 = a.get_potential_energy()
    n2 = opt.nsteps

    assert e2 == pytest.approx(e1, abs=1e-6)
    assert n2 < n1
    assert all(np.array(reset_history) > 0)
