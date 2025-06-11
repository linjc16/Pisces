import math

import pytest

from ase import Atoms
from ase.build import molecule
from ase.utils.ff import VdW, get_vdw_potential_value


@pytest.fixture()
def atoms() -> Atoms:
    mol = molecule("H2")
    mol.center(5.0)
    return mol


def test_utils_ff_vdw(atoms: Atoms) -> None:
    """Check the equivalent van der Waals energies."""
    e1, e2, s1, s2, scale = 0.646, 1.557, 0.777, 0.724, 1.2236

    vdw_a = VdW(
        0, 1, epsiloni=e1, epsilonj=e2, sigmai=s1, sigmaj=s2, scale=scale
    )

    vdw_p = VdW(
        0, 1, epsilonij=math.sqrt(e1 * e2), sigmaij=(s1 + s2) / 2.0, scale=scale
    )

    e_ref = get_vdw_potential_value(atoms, vdw_a)
    assert get_vdw_potential_value(atoms, vdw_p) == pytest.approx(e_ref)
