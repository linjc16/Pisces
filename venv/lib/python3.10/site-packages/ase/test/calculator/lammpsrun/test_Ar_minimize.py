import pytest
from numpy.testing import assert_allclose

from ase.calculators.fd import calculate_numerical_forces
from ase.cluster.icosahedron import Icosahedron
from ase.data import atomic_masses, atomic_numbers
from ase.optimize import LBFGS


@pytest.fixture()
def ar_nc():
    ar_nc = Icosahedron('Ar', noshells=2)
    ar_nc.cell = [[300, 0, 0], [0, 300, 0], [0, 0, 300]]
    ar_nc.pbc = True
    return ar_nc


@pytest.fixture()
def params():
    params = {}
    params['pair_style'] = 'lj/cut 8.0'
    params['pair_coeff'] = ['1 1 0.0108102 3.345']
    params['masses'] = ['1 {}'.format(atomic_masses[atomic_numbers['Ar']])]
    return params


@pytest.mark.calculator_lite()
@pytest.mark.calculator('lammpsrun')
def test_Ar_minimize(factory, ar_nc, params):
    with factory.calc(specorder=['Ar'], **params) as calc:
        ar_nc.calc = calc

        assert_allclose(ar_nc.get_potential_energy(), -0.468147667942117,
                        atol=1e-4, rtol=1e-4)
        assert_allclose(ar_nc.get_forces(),
                        calculate_numerical_forces(ar_nc),
                        atol=1e-4, rtol=1e-4)

        with LBFGS(ar_nc) as dyn:
            dyn.run(fmax=1E-6)

        assert_allclose(ar_nc.get_potential_energy(), -0.4791815886953914,
                        atol=1e-4, rtol=1e-4)
        assert_allclose(ar_nc.get_forces(),
                        calculate_numerical_forces(ar_nc),
                        atol=1e-4, rtol=1e-4)


@pytest.mark.calculator_lite()
@pytest.mark.calculator('lammpsrun')
def test_Ar_minimize_multistep(factory, ar_nc, params):
    ar_nc = Icosahedron('Ar', noshells=2)
    ar_nc.cell = [[300, 0, 0], [0, 300, 0], [0, 0, 300]]
    ar_nc.pbc = True

    with factory.calc(specorder=['Ar'], **params) as calc:
        ar_nc.calc = calc
        F1_numer = calculate_numerical_forces(ar_nc)

        assert_allclose(ar_nc.get_potential_energy(), -0.468147667942117,
                        atol=1e-4, rtol=1e-4)
        assert_allclose(ar_nc.get_forces(), F1_numer,
                        atol=1e-4, rtol=1e-4)

        calc.set(minimize='1.0e-15 1.0e-6 2000 4000')

        # set_atoms=True to read final coordinates after minimization
        calc.run(set_atoms=True)

        # get final coordinates after minimization
        ar_nc.set_positions(calc.atoms.positions)

        assert_allclose(ar_nc.get_potential_energy(), -0.4791815887032201,
                        atol=1e-4, rtol=1e-4)
        assert_allclose(ar_nc.get_forces(),
                        calculate_numerical_forces(ar_nc),
                        atol=1e-4, rtol=1e-4)
