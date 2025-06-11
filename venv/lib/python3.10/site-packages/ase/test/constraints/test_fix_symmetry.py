import numpy as np
import pytest

from ase.atoms import Atoms
from ase.build import bulk
from ase.calculators.calculator import all_changes
from ase.calculators.lj import LennardJones
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter, UnitCellFilter
from ase.md.verlet import VelocityVerlet
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.spacegroup.symmetrize import check_symmetry, is_subgroup

spglib = pytest.importorskip('spglib')


pytestmark = pytest.mark.optimize


class NoisyLennardJones(LennardJones):
    def __init__(self, *args, rng=None, **kwargs):
        self.rng = rng
        LennardJones.__init__(self, *args, **kwargs)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        LennardJones.calculate(self, atoms, properties, system_changes)
        if 'forces' in self.results:
            self.results['forces'] += 1e-4 * self.rng.normal(
                size=self.results['forces'].shape, )
        if 'stress' in self.results:
            self.results['stress'] += 1e-4 * self.rng.normal(
                size=self.results['stress'].shape, )


def setup_cell():
    # setup an bcc Al cell
    at_init = bulk('Al', 'bcc', a=2 / np.sqrt(3), cubic=True)

    F = np.eye(3)
    for k in range(3):
        L = list(range(3))
        L.remove(k)
        (i, j) = L
        R = np.eye(3)
        theta = 0.1 * (k + 1)
        R[i, i] = np.cos(theta)
        R[j, j] = np.cos(theta)
        R[i, j] = np.sin(theta)
        R[j, i] = -np.sin(theta)
        F = np.dot(F, R)
    at_rot = at_init.copy()
    at_rot.set_cell(at_rot.cell @ F, True)
    return at_init, at_rot


def symmetrized_optimisation(at_init, filter):
    rng = np.random.RandomState(1)
    at = at_init.copy()
    at.calc = NoisyLennardJones(rng=rng)

    at_cell = filter(at)
    print("Initial Energy", at.get_potential_energy(), at.get_volume())
    with PreconLBFGS(at_cell, precon=None) as dyn:
        dyn.run(steps=300, fmax=0.001)
        print("n_steps", dyn.get_number_of_steps())
    print("Final Energy", at.get_potential_energy(), at.get_volume())
    print("Final forces\n", at.get_forces())
    print("Final stress\n", at.get_stress())

    print("initial symmetry at 1e-6")
    di = check_symmetry(at_init, 1.0e-6, verbose=True)
    print("final symmetry at 1e-6")
    df = check_symmetry(at, 1.0e-6, verbose=True)
    return di, df


def test_as_dict():
    atoms = bulk("Cu")
    atoms.set_constraint(FixSymmetry(atoms))
    assert atoms.constraints[0].todict() == {
        'name': 'FixSymmetry',
        'kwargs': {
            'atoms': bulk('Cu'),
            'symprec': 0.01,
            'adjust_positions': True,
            'adjust_cell': True,
            'verbose': False,
        },
    }


def test_fail_md():
    atoms = bulk("Cu")
    atoms.set_constraint(FixSymmetry(atoms))

    atoms.calc = LennardJones()
    # This will not fail if the user has no logfile specified
    # a little bit weird...
    with pytest.raises(NotImplementedError):
        dyn = VelocityVerlet(atoms, timestep=1.0, logfile="-")
        dyn.run(5)


@pytest.fixture(params=[UnitCellFilter, FrechetCellFilter])
def filter(request):
    return request.param


@pytest.mark.filterwarnings('ignore:ASE Atoms-like input is deprecated')
@pytest.mark.filterwarnings('ignore:Armijo linesearch failed')
def test_no_symmetrization(filter):
    at_init, _at_rot = setup_cell()
    at_unsym = at_init.copy()
    di, df = symmetrized_optimisation(at_unsym, filter)
    assert di.number == 229 and not is_subgroup(sub_data=di, sup_data=df)


@pytest.mark.filterwarnings('ignore:ASE Atoms-like input is deprecated')
@pytest.mark.filterwarnings('ignore:Armijo linesearch failed')
def test_no_sym_rotated(filter):
    _at_init, at_rot = setup_cell()
    at_unsym_rot = at_rot.copy()
    di, df = symmetrized_optimisation(at_unsym_rot, filter)
    assert di.number == 229 and not is_subgroup(sub_data=di, sup_data=df)


@pytest.mark.filterwarnings('ignore:ASE Atoms-like input is deprecated')
@pytest.mark.filterwarnings('ignore:Armijo linesearch failed')
def test_sym_adj_cell(filter):
    at_init, _at_rot = setup_cell()
    at_sym_3 = at_init.copy()
    at_sym_3.set_constraint(
        FixSymmetry(at_sym_3, adjust_positions=True, adjust_cell=True))
    di, df = symmetrized_optimisation(at_sym_3, filter)
    assert di.number == 229 and is_subgroup(sub_data=di, sup_data=df)


@pytest.mark.filterwarnings('ignore:ASE Atoms-like input is deprecated')
@pytest.mark.filterwarnings('ignore:Armijo linesearch failed')
def test_sym_rot_adj_cell(filter):
    at_init, _at_rot = setup_cell()
    at_sym_3_rot = at_init.copy()
    at_sym_3_rot.set_constraint(
        FixSymmetry(at_sym_3_rot, adjust_positions=True, adjust_cell=True))
    di, df = symmetrized_optimisation(at_sym_3_rot, filter)
    assert di.number == 229 and is_subgroup(sub_data=di, sup_data=df)


@pytest.mark.filterwarnings('ignore:ASE Atoms-like input is deprecated')
def test_fix_symmetry_shuffle_indices():
    atoms = Atoms(
        'AlFeAl6', cell=[6] * 3,
        positions=[[0, 0, 0], [2.9, 2.9, 2.9], [0, 0, 3], [0, 3, 0],
                   [0, 3, 3], [3, 0, 0], [3, 0, 3], [3, 3, 0]], pbc=True)
    atoms.set_constraint(FixSymmetry(atoms))
    at_permut = atoms[[0, 2, 3, 4, 5, 6, 7, 1]]
    pos0 = atoms.get_positions()

    def perturb(atoms, pos0, at_i, dpos):
        positions = pos0.copy()
        positions[at_i] += dpos
        atoms.set_positions(positions)
        new_p = atoms.get_positions()
        return pos0[at_i] - new_p[at_i]

    dp1 = perturb(atoms, pos0, 1, (0.0, 0.1, -0.1))
    dp2 = perturb(atoms, pos0, 2, (0.0, 0.1, -0.1))
    pos0 = at_permut.get_positions()
    permut_dp1 = perturb(at_permut, pos0, 7, (0.0, 0.1, -0.1))
    permut_dp2 = perturb(at_permut, pos0, 1, (0.0, 0.1, -0.1))
    assert np.max(np.abs(dp1 - permut_dp1)) < 1.0e-10
    assert np.max(np.abs(dp2 - permut_dp2)) < 1.0e-10
