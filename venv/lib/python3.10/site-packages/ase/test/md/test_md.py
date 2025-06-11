import pytest

from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import Trajectory
from ase.md import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


@pytest.fixture(name="atoms")
def fixture_atoms():
    atoms = bulk("Au") * 2
    MaxwellBoltzmannDistribution(atoms, temperature_K=100.0)
    atoms.calc = EMT()
    return atoms


def test_md(atoms, testdir):
    def f():
        print(atoms.get_potential_energy(), atoms.get_total_energy())

    with VelocityVerlet(atoms, timestep=0.1) as md:
        md.attach(f)
        with Trajectory('Cu2.traj', 'w', atoms) as traj:
            md.attach(traj.write, interval=3)
            md.run(steps=20)

    with Trajectory('Cu2.traj', 'r') as traj:
        traj[-1]

    # Really?? No assertion at all?


@pytest.mark.parametrize("md_class", [VelocityVerlet])
def test_run_twice(md_class, atoms):
    """Test if `steps` increments `max_steps` when `run` is called twice."""
    steps = 5
    with md_class(atoms, timestep=1.0 * units.fs) as md:
        md.run(steps=steps)
        md.run(steps=steps)
    assert md.nsteps == 2 * steps
    assert md.max_steps == 2 * steps
