import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.filters import StrainFilter
from ase.optimize.mdmin import MDMin


@pytest.mark.optimize()
def test_strain_emt():
    cu = bulk('Cu', 'fcc', a=3.6)
    cu.calc = EMT()
    f = StrainFilter(cu)
    opt = MDMin(f, dt=0.01)
    opt.run(0.1, steps=2)
