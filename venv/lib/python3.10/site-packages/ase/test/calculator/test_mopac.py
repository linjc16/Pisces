import pytest
from numpy.testing import assert_allclose

from ase.build import molecule
from ase.calculators.mopac import MOPAC
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS


@pytest.mark.calculator_lite()
@pytest.mark.calculator()
def test_mopac(mopac_factory):
    """Test H2 molecule atomization with MOPAC."""
    # Unrestricted Hartree-Fock; enable magmom calc
    h2 = molecule('H2',
                  calculator=mopac_factory.calc(label='h2',
                                                task='1SCF GRADIENTS UHF'))
    with Trajectory('h2.traj', mode='w') as traj:
        BFGS(h2, trajectory=traj).run(fmax=0.01)
    e2 = h2.get_potential_energy()
    h1 = h2.copy()
    del h1[1]
    h1.set_initial_magnetic_moments([1])
    h1.calc = mopac_factory.calc(label='h1')
    e1 = h1.get_potential_energy()
    d = h2.get_distance(0, 1)
    ea = 2 * e1 - e2
    print(d, ea)
    assert abs(d - 0.759) < 0.001
    assert abs(ea - 5.907) < 0.001
    h2o = molecule('H2O', calculator=mopac_factory.calc(label='h2o',
                                                        tasks='GRADIENTS'))
    h2o.get_potential_energy()
    print('dipole:', h2o.get_dipole_moment())
    atoms = MOPAC.read_atoms('h2', profile=h2o.calc.profile)
    print('magmom:', atoms.calc.get_magnetic_moment())
    print('PM7 homo lumo:', atoms.calc.get_homo_lumo_levels())
    atoms.calc.set(method='AM1')
    atoms.get_potential_energy()
    print('AM1 homo lumo:', atoms.calc.get_homo_lumo_levels())
    calc = mopac_factory.calc(restart='h1')
    print('magmom:', calc.get_magnetic_moment())


@pytest.mark.calculator_lite()
def test_mopac_forces_consistent(mopac_factory):
    """Check MOPAC forces follow Newton's 3rd Law"""

    ch4 = molecule('CH4')
    ch4.rattle()
    ch4.calc = mopac_factory.calc(task='1SCF GRADIENTS', method='PM7')
    forces = ch4.get_forces()

    assert_allclose(forces.sum(axis=0), [0, 0, 0], atol=1e-7)
