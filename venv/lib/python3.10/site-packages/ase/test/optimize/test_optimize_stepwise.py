import pytest

from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS


@pytest.fixture()
def atoms0():
    atoms = bulk('Au', cubic=True)
    atoms.rattle(stdev=0.15)
    return atoms


@pytest.fixture()
def atoms(atoms0):
    atoms = atoms0.copy()
    atoms.calc = EMT()
    return atoms


@pytest.mark.optimize()
def test_irun_start(atoms0, atoms):
    opt = BFGS(atoms)
    irun = opt.irun(fmax=0.0)
    next(irun)  # Initially it yields without yet having performed a step:
    assert not compare_atoms(atoms0, atoms)
    next(irun)  # Now it must have performed a step:
    assert compare_atoms(atoms0, atoms) == ['positions']


@pytest.mark.optimize()
def test_attach_trajectory(tmp_path, atoms0, atoms):
    # Previously one needed to specify the same atoms object to
    # both trajectory and optimizer, if trajectory was opened separately.
    # This test ensures that doing so is no longer necessary.
    #
    # We also opportunistically test that the first image written to the
    # trajectory is in fact the original atoms object.

    trajfile = tmp_path / 'tmp.traj'
    n_yields = 3
    with Trajectory(trajfile, 'w') as traj:
        opt = BFGS(atoms, trajectory=traj)
        irun = opt.irun(fmax=0.0)

        for _ in range(n_yields):
            next(irun)

    with Trajectory(tmp_path / 'tmp.traj') as traj:
        images = list(traj)

    # The number of images must be equal to the number of yielding.
    assert len(images) == n_yields

    assert not compare_atoms(atoms0, images[0])
    assert compare_atoms(atoms0, images[-1]) == ['positions']
