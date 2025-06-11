from math import sqrt

import pytest

from ase.build import bulk
from ase.filters import StrainFilter
from ase.optimize.mdmin import MDMin

a = 3.6


@pytest.mark.optimize()
def test_strain_fcc(asap3):
    cu = bulk('Cu', a=a) * (6, 6, 6)
    cu.calc = asap3.EMT()
    f = StrainFilter(cu, [1, 1, 1, 0, 0, 0])
    opt = MDMin(f, dt=0.01)
    opt.run(0.001)


@pytest.mark.optimize()
def test_strain_hcp(asap3):
    cu = bulk('Cu', 'hcp', a=a / sqrt(2))
    cu.cell[1, 0] -= 0.05
    cu *= (6, 6, 3)

    cu.calc = asap3.EMT()
    f = StrainFilter(cu)
    opt = MDMin(f, dt=0.01)
    opt.run(0.01)
