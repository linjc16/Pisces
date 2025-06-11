import numpy as np
import pytest

from ase.build import bulk, molecule
from ase.calculators.emt import EMT
from ase.filters import (
    ExpCellFilter,
    Filter,
    FrechetCellFilter,
    StrainFilter,
    UnitCellFilter,
)
from ase.optimize import QuasiNewton
from ase.stress import full_3x3_to_voigt_6_strain, voigt_6_to_full_3x3_strain


@pytest.mark.optimize()
def test_filter(testdir):
    """Test that the filter and trajectories are playing well together."""

    atoms = molecule('CO2')
    atoms.calc = EMT()
    filter = Filter(atoms, indices=[1, 2])

    with QuasiNewton(filter, trajectory='filter-test.traj',
                     logfile='filter-test.log') as opt:
        opt.run()
    # No assertions=??


@pytest.mark.optimize()
@pytest.mark.filterwarnings("ignore:Use FrechetCellFilter")
@pytest.mark.parametrize(
    'filterclass', [StrainFilter,
                    UnitCellFilter,
                    FrechetCellFilter,
                    ExpCellFilter])
@pytest.mark.parametrize(
    'mask', [[1, 1, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0],
             [1, 0, 1, 1, 0, 0],
             [0, 1, 0, 0, 1, 1]]
)
def test_apply_strain_to_mask(filterclass, mask):
    cu = bulk('Cu', a=3.14) * (6, 3, 1)
    orig_cell = cu.cell.copy()
    rng = np.random.RandomState(69)

    # Create extreme deformation
    deformation_vv = np.eye(3) + 1e2 * rng.randn(3, 3)
    filter = filterclass(cu, mask=mask)
    if filterclass is not StrainFilter:
        pos_and_deform = \
            np.concatenate((cu.get_positions(),
                            deformation_vv), axis=0)
    else:
        pos_and_deform = full_3x3_to_voigt_6_strain(deformation_vv)

    # Apply the deformation to the filter, which should then apply it
    # with a mask.
    filter.set_positions(pos_and_deform)
    full_mask = voigt_6_to_full_3x3_strain(mask) != 0

    # Ensure the mask is respected to a very tight numerical tolerance
    assert np.linalg.solve(orig_cell, cu.cell)[~full_mask] == \
        pytest.approx(np.eye(3)[~full_mask], abs=1e-12)
