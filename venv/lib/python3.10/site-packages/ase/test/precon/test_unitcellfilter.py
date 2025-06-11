import pytest

import ase
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.filters import Filter, FrechetCellFilter, UnitCellFilter
from ase.optimize.precon import Exp, PreconLBFGS


# @pytest.mark.skip('FAILS WITH PYAMG')
@pytest.mark.optimize()
@pytest.mark.slow()
# Ignore UserWarning by failure of Armijo line search in PreconLBFGS
@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize("filter_cls, tol", [
    (UnitCellFilter, 1e-3),
    # FrechetCellFilter allows relaxing to lower tolerance
    (FrechetCellFilter, 1e-7),
])
def test_precon(filter_cls, tol: float):
    cu0: ase.Atoms = bulk("Cu") * (2, 2, 2)
    lj = LennardJones(sigma=cu0.get_distance(0, 1))

    ratio = 1.2
    cu = cu0.copy()
    cu.set_cell(ratio * cu.get_cell())
    cu.calc = lj
    filter: Filter = filter_cls(cu, constant_volume=True)
    opt = PreconLBFGS(filter, precon=Exp(mu=1.0, mu_c=1.0))
    opt.run(fmax=1e-3)
    assert abs(cu.get_volume() / cu0.get_volume() - ratio**3) < tol
