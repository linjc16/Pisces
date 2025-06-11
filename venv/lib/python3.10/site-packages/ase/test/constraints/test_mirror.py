import pytest

from ase.build import molecule
from ase.calculators.emt import EMT
from ase.constraints import (
    ExternalForce,
    FixBondLength,
    MirrorForce,
    MirrorTorque,
)
from ase.optimize import FIRE


@pytest.mark.optimize()
def test_mirror():
    atoms = molecule('cyclobutene')
    dist = atoms.get_distance(0, 1)
    con1 = MirrorForce(2, 3, max_dist=5., fmax=0.05)
    con2 = FixBondLength(0, 1)
    atoms.set_constraint([con1, con2])
    atoms.calc = EMT()
    opt = FIRE(atoms)
    opt.run(fmax=0.05)
    assert round(dist - atoms.get_distance(0, 1), 5) == 0

    atoms = molecule('butadiene')
    # Break symmetry
    atoms[0].position[2] += 0.2
    dist = atoms.get_distance(1, 2)
    con1 = MirrorTorque(0, 1, 2, 3, fmax=0.05)
    con2 = ExternalForce(9, 4, f_ext=0.1)
    atoms.set_constraint([con1, con2])
    atoms.calc = EMT()
    opt = FIRE(atoms)
    opt.run(fmax=0.05, steps=300)
    # The result is not realistic because of EMT
