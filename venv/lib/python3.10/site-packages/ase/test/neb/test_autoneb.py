from pathlib import Path

import pytest

from ase import Atoms
from ase.build import add_adsorbate, fcc211
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.mep import AutoNEB, NEBTools
from ase.optimize import BFGS, QuasiNewton


@pytest.mark.optimize()
def test_autoneb(asap3, testdir):
    EMT = asap3.EMT
    fmax = 0.02

    # Pt atom adsorbed in a hollow site:
    slab = fcc211('Pt', size=(3, 2, 2), vacuum=4.0)
    add_adsorbate(slab, 'Pt', 0.5, (-0.1, 2.7))

    # Fix second and third layers:
    slab.set_constraint(FixAtoms(range(6, 12)))

    # Use EMT potential:
    slab.calc = EMT()

    # Initial state:
    with QuasiNewton(slab, trajectory='neb000.traj') as qn:
        qn.run(fmax=fmax)

    # Final state:
    slab[-1].x += slab.get_cell()[0, 0]
    slab[-1].y += 2.8
    with QuasiNewton(slab, trajectory='neb001.traj') as qn:
        qn.run(fmax=fmax)

    # Stops PermissionError on Win32 for access to
    # the traj file that remains open.
    del qn

    def attach_calculators(images):
        for i in range(len(images)):
            images[i].calc = EMT()

    autoneb = AutoNEB(attach_calculators,
                      prefix='neb',
                      optimizer=BFGS,
                      n_simul=3,
                      n_max=7,
                      fmax=fmax,
                      k=0.5,
                      parallel=False,
                      maxsteps=[50, 1000])
    autoneb.run()

    nebtools = NEBTools(autoneb.all_images)
    assert abs(nebtools.get_barrier()[0] - 0.937) < 1e-3


@pytest.mark.optimize()
def test_Au2Ag(testdir):
    def attach_calculators(images):
        for i in range(len(images)):
            images[i].calc = EMT()

    d = 4
    initial = Atoms('Au2Ag',
                    positions=((-d, 0, 0), (-d / 2, 0, 0), (d, 0, 0)))

    middle = initial.copy()
    middle[1].position[0] = 0

    final = initial.copy()
    final[1].position[0] += d

    attach_calculators([initial, middle, final])
    for i, image in enumerate([initial, middle, final]):
        image.set_constraint(FixAtoms([0, 2]))

    fmax = 0.05
    for i, image in enumerate([initial, final]):
        opt = QuasiNewton(image)
        opt.run(fmax=fmax)
    middle.get_forces()

    prefix = Path('subdir') / 'neb'
    prefix.parent.mkdir()
    for i, image in enumerate([initial, middle, final]):
        image.write(f'{prefix}00{i}.traj')

    autoneb = AutoNEB(attach_calculators,
                      prefix=prefix,
                      optimizer=QuasiNewton,
                      n_simul=1,
                      n_max=5,
                      fmax=fmax,
                      k=0.5,
                      parallel=False,
                      maxsteps=[20, 1000])
    autoneb.run()

    nebtools = NEBTools(autoneb.all_images)
    assert nebtools.get_barrier()[0] == pytest.approx(4.185, 1e-3)
