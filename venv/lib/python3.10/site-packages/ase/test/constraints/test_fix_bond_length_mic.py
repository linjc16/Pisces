import pytest

import ase
from ase.calculators.lj import LennardJones
from ase.constraints import FixBondLength
from ase.optimize import FIRE


@pytest.mark.optimize()
@pytest.mark.parametrize('wrap', [False, True])
def test_fix_bond_length_mic(wrap):
    a = ase.Atoms('CCC',
                  positions=[[1, 0, 5],
                             [0, 1, 5],
                             [-1, 0.5, 5]],
                  cell=[10, 10, 10],
                  pbc=True)

    if wrap:
        a.set_scaled_positions(a.get_scaled_positions() % 1.0)
    a.calc = LennardJones()
    a.set_constraint(FixBondLength(0, 2))

    assert a.get_number_of_degrees_of_freedom() == len(a) * 3 - 1

    d1 = a.get_distance(0, 2, mic=True)

    with FIRE(a, logfile=None) as opt:
        opt.run(fmax=0.01)
    e = a.get_potential_energy()
    d2 = a.get_distance(0, 2, mic=True)
    assert abs(e - -2.034988) < 1e-6
    assert abs(d1 - d2) < 1e-6
