import numpy as np
import pytest
from scipy.optimize import check_grad

from ase import Atoms
from ase.build import bulk
from ase.calculators.fd import (
    calculate_numerical_forces,
    calculate_numerical_stress,
)
from ase.calculators.morse import MorsePotential, fcut, fcut_d
from ase.vibrations import Vibrations

De = 5.
Re = 3.
rho0 = 2.


def test_gs_minimum_energy():
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, Re]])
    atoms.calc = MorsePotential(epsilon=De, r0=Re)
    assert atoms.get_potential_energy() == -De


def test_gs_vibrations(testdir):
    # check ground state vibrations
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, Re]])
    atoms.calc = MorsePotential(epsilon=De, r0=Re, rho0=rho0)
    vib = Vibrations(atoms)
    vib.run()


def test_cutoff():
    # check that fcut_d is the derivative of fcut
    r1 = 2.0
    r2 = 3.0
    r = np.linspace(r1 - 0.5, r2 + 0.5, 100)
    for R in r:
        assert check_grad(fcut, fcut_d, np.array([R]), r1, r2) < 1e-5


def test_forces_and_stress():
    atoms = bulk('Cu', cubic=True)
    atoms.calc = MorsePotential(A=4.0, epsilon=1.0, r0=2.55)
    atoms.rattle(0.1)

    forces = atoms.get_forces()
    numerical_forces = calculate_numerical_forces(atoms, eps=1e-5)
    np.testing.assert_allclose(forces, numerical_forces, atol=1e-5)

    stress = atoms.get_stress()
    numerical_stress = calculate_numerical_stress(atoms, eps=1e-5)
    np.testing.assert_allclose(stress, numerical_stress, atol=1e-5)


def fake_neighbor_list(*args, **kwargs):
    raise RuntimeError('test_neighbor_list')


def test_override_neighbor_list():
    with pytest.raises(RuntimeError, match='test_neighbor_list'):
        atoms = bulk('Cu', cubic=True)
        atoms.calc = MorsePotential(A=4.0, epsilon=1.0, r0=2.55,
                                    neighbor_list=fake_neighbor_list)
        _ = atoms.get_potential_energy()
