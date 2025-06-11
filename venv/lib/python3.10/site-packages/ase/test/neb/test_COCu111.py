from math import sqrt

import pytest

from ase import Atom, Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import Trajectory
from ase.mep import NEB
from ase.optimize import BFGS, QuasiNewton


@pytest.mark.optimize()
def test_COCu111(testdir):
    # Distance between Cu atoms on a (111) surface:
    a = 3.6
    d = a / sqrt(2)
    fcc111 = Atoms(symbols='Cu',
                   cell=[(d, 0, 0),
                         (d / 2, d * sqrt(3) / 2, 0),
                         (d / 2, d * sqrt(3) / 6, -a / sqrt(3))],
                   pbc=True)
    slab = fcc111 * (2, 2, 2)
    slab.set_cell([2 * d, d * sqrt(3), 1])
    slab.set_pbc((1, 1, 0))
    slab.calc = EMT()
    Z = slab.get_positions()[:, 2]
    indices = [i for i, z in enumerate(Z) if z < Z.mean()]
    constraint = FixAtoms(indices=indices)
    slab.set_constraint(constraint)
    with QuasiNewton(slab) as dyn:
        dyn.run(fmax=0.05)
    Z = slab.get_positions()[:, 2]
    print(Z[0] - Z[1])
    print(Z[1] - Z[2])
    print(Z[2] - Z[3])

    b = 1.2
    h = 1.5
    slab += Atom('C', (d / 2, -b / 2, h))
    slab += Atom('O', (d / 2, +b / 2, h))
    with QuasiNewton(slab) as dyn:
        dyn.run(fmax=0.05)

    # Make band:
    images = [slab]
    for _ in range(4):
        image = slab.copy()
        # Set constraints and calculator:
        image.set_constraint(constraint)
        image.calc = EMT()
        images.append(image)

    # Displace last image:
    image = images[-1]
    image[-2].position = image[-1].position
    image[-1].x = d
    image[-1].y = d / sqrt(3)

    with QuasiNewton(images[-1]) as dyn:
        dyn.run(fmax=0.05)
    neb = NEB(images, climb=not True)

    # Interpolate positions between initial and final states:
    neb.interpolate(method='idpp')

    for image in images:
        print(image.positions[-1], image.get_potential_energy())

    with BFGS(neb, maxstep=0.04, trajectory='mep.traj') as dyn:
        dyn.run(fmax=0.1)

    for image in images:
        print(image.positions[-1], image.get_potential_energy())

    # Trying to read description of optimization from trajectory
    with Trajectory('mep.traj') as traj:
        assert traj.description['optimizer'] == 'BFGS'
        for key, value in traj.description.items():
            print(key, value)
        print(traj.ase_version)
