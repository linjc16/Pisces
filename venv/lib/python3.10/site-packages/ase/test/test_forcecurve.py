from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import read
from ase.md import VelocityVerlet
from ase.units import fs
from ase.utils.forcecurve import force_curve


def test_forcecurve(testdir, plt):
    atoms = bulk('Au', cubic=True) * (2, 1, 1)
    atoms.calc = EMT()
    atoms.rattle(stdev=0.05)

    with VelocityVerlet(atoms, timestep=12.0 * fs,
                        trajectory='tmp.traj') as md:
        md.run(steps=10)
    images = read('tmp.traj', ':')
    force_curve(images)

    # import pylab as plt
    # plt.show()
