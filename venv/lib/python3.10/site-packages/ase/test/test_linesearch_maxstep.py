import numpy as np
import pytest

from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS, BFGSLineSearch
from ase.optimize.precon import Exp, PreconLBFGS


@pytest.fixture()
def positions():
    pos = np.array([
        [5.8324672234339969, 8.5510800490537271, 5.686535793302002],
        [7.3587688835494625, 5.646353802990923, 6.8378173997818958],
        [7.9908510609316235, 5.4456005797117335, 4.5249260246251213],
        [9.7103024117445145, 6.4768915365291466, 4.6502022197421278],
        [9.5232482249292509, 8.7417754382952051, 4.6747936030744448],
        [8.2738330473112036, 7.640248516254645, 6.1624124370797215],
        [7.4198265919217921, 9.2882534361810016, 4.3654132356242874],
        [6.8506783463494623, 9.2004422130272605, 8.611538688631887],
        [5.9081131977596133, 5.6951755645279949, 5.4134092632199602],
        [9.356736354387575, 9.2718534012646359, 8.491942486888524],
        [9.0390271264592403, 9.5752757925665453, 6.4771649275571779],
        [7.0554382804264533, 7.0016335250680779, 8.418151938177477],
        [9.4855926945401272, 5.5650406772147694, 6.8445655410690591],
    ])
    return pos


@pytest.fixture()
def atoms(positions):
    atoms = Atoms('Pt13', positions=positions, cell=[15] * 3)
    atoms.calc = EMT()
    return atoms


labels = [
    'BFGS',
    'BFGSLineSearch',
    'PreconLBFGS_Armijo',  # XXXX FAILS WITH PYAMG
    'PreconLBFGS_Wolff'    # XXXXX FAILS WITH PYAMG
]
optimizers = [BFGS, BFGSLineSearch, PreconLBFGS, PreconLBFGS]


@pytest.mark.optimize()
@pytest.mark.parametrize(
    'optcls, name',
    zip(optimizers, labels),
)
def test_linesearch(optcls, name, atoms, positions):
    maxstep = 0.2
    kwargs = {'maxstep': maxstep, 'logfile': None}
    if 'Precon' in name:
        kwargs['precon'] = Exp(A=3)
        kwargs['use_armijo'] = 'Armijo' in name

    with optcls(atoms, **kwargs) as opt:
        opt.run(steps=1)

    dr = atoms.get_positions() - positions
    steplengths = (dr**2).sum(1)**0.5
    longest_step = np.max(steplengths)
    assert longest_step < maxstep + 1e-8
