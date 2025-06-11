import numpy as np

from ase.build import bulk, molecule
from ase.calculators.gulp import GULP
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS


def test_gulp_opt(gulp_factory):
    # GULP optmization test
    atoms = molecule('H2O')
    atoms1 = atoms.copy()
    atoms1.calc = GULP(library='reaxff.lib')
    with BFGS(atoms1) as opt1:
        opt1.run(fmax=0.005)

    atoms2 = atoms.copy()
    calc2 = GULP(keywords='opti conp', library='reaxff.lib')
    with calc2.get_optimizer(atoms2) as opt2:
        opt2.run()

    print(np.abs(opt1.atoms.positions - opt2.atoms.positions))
    assert np.abs(opt1.atoms.positions - opt2.atoms.positions).max() < 1e-5

    # GULP optimization test using stress
    atoms = bulk('Au', 'bcc', a=2.7, cubic=True)
    atoms1 = atoms.copy()
    atoms1.calc = GULP(keywords='conp gradient stress_out',
                       library='reaxff_general.lib')
    atoms1f = FrechetCellFilter(atoms1)
    with BFGS(atoms1f) as opt1:
        opt1.run(fmax=0.005)

    atoms2 = atoms.copy()
    calc2 = GULP(keywords='opti conp', library='reaxff_general.lib')
    with calc2.get_optimizer(atoms2) as opt2:
        opt2.run()

    print(np.abs(opt1.atoms.positions - opt2.atoms.positions))
    assert np.abs(opt1.atoms.positions - opt2.atoms.positions).max() < 1e-5
