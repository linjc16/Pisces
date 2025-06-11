import numpy as np
import pytest

from ase.build import molecule
from ase.calculators.ff import ForceField
from ase.optimize.precon import FF
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.optimize.precon.neighbors import get_neighbours
from ase.utils.ff import Angle, Dihedral, Morse, VdW


@pytest.fixture(scope='module')
def atoms0():
    a = molecule('C60')
    a.set_cell(50.0 * np.identity(3))
    return a


@pytest.fixture(scope='module')
def forcefield_params(atoms0):
    # force field parameters for fulleren, Z. Berkai at al.
    # Energy Procedia, 74, 2015, 59-64
    a = atoms0
    cutoff = 1.5
    morse_D = 6.1322
    morse_alpha = 1.8502
    morse_r0 = 1.4322
    angle_k = 10.0
    angle_a0 = np.deg2rad(120.0)
    dihedral_k = 0.346
    vdw_epsilonij = 0.0115
    vdw_rminij = 3.4681

    neighbor_list = [[] for _ in range(len(a))]
    vdw_list = np.ones((len(a), len(a)), dtype=bool)
    morses = []
    angles = []
    dihedrals = []
    vdws = []

    # create neighbor list
    i_list, j_list, _d_list, _fixed_atoms = get_neighbours(
        atoms=a, r_cut=cutoff)

    for i, j in zip(i_list, j_list):
        neighbor_list[i].append(j)
    for neighbor in neighbor_list:
        neighbor.sort()

    # create lists of morse, bending and torsion interactions
    for i in range(len(a)):
        for jj in range(len(neighbor_list[i])):
            j = neighbor_list[i][jj]
            if j > i:
                morses.append(Morse(atomi=i, atomj=j, D=morse_D,
                                    alpha=morse_alpha, r0=morse_r0))
            vdw_list[i, j] = vdw_list[j, i] = False
            for kk in range(jj + 1, len(neighbor_list[i])):
                k = neighbor_list[i][kk]
                angles.append(Angle(atomi=j, atomj=i, atomk=k, k=angle_k,
                                    a0=angle_a0, cos=True))
                vdw_list[j, k] = vdw_list[k, j] = False
                for ll in range(kk + 1, len(neighbor_list[i])):
                    L = neighbor_list[i][ll]
                    dihedrals.append(Dihedral(atomi=j, atomj=i, atomk=k,
                                              atoml=L,
                                              k=dihedral_k))

    # create list of van der Waals interactions
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if vdw_list[i, j]:
                vdws.append(VdW(atomi=i, atomj=j, epsilonij=vdw_epsilonij,
                                rminij=vdw_rminij))

    return dict(morses=morses, angles=angles, dihedrals=dihedrals, vdws=vdws)


@pytest.fixture()
def calc(forcefield_params):
    return ForceField(**forcefield_params)


@pytest.fixture()
def atoms(atoms0, calc):
    atoms = atoms0.copy()
    atoms.calc = calc
    atoms.rattle(0.05)
    return atoms


ref_energy = 17.238525

# @pytest.mark.skip('FAILS WITH PYAMG')


@pytest.mark.optimize()
@pytest.mark.slow()
def test_opt_with_precon(atoms, forcefield_params):
    kw = dict(forcefield_params)
    kw.pop('vdws')
    precon = FF(**kw)
    opt = PreconLBFGS(atoms, use_armijo=True, precon=precon)
    opt.run(fmax=0.1)
    e = atoms.get_potential_energy()
    assert abs(e - ref_energy) < 0.01
