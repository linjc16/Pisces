# type: ignore
import numpy as np

from ase.build import molecule
from ase.calculators.turbomole import Turbomole


def test_turbomole_h2o(turbomole_factory):
    mol = molecule('H2O')

    params = {
        'title': 'water',
        'task': 'geometry optimization',
        'use redundant internals': True,
        'basis set name': 'def2-SV(P)',
        'total charge': 0,
        'multiplicity': 1,
        'use dft': True,
        'density functional': 'b3-lyp',
        'use resolution of identity': True,
        'ri memory': 1000,
        'force convergence': 0.001,
        'geometry optimization iterations': 50,
        'scf iterations': 100
    }

    calc = Turbomole(**params)
    mol.calc = calc
    calc.calculate(mol)
    assert calc.converged

    # use the get_property() method
    energy = calc.get_property('energy', mol, False)
    assert energy is not False
    assert np.isclose(energy, -2076.286138, atol=1e-5)
    forces = calc.get_property('forces', mol, False)
    assert forces is not False
    assert np.linalg.norm(forces) < 0.01
    dipole = np.linalg.norm(calc.get_property('dipole', mol, False))
    assert np.isclose(dipole, 0.448, rtol=0.01)

    # use the get_results() method
    results = calc.get_results()
    print(results['molecular orbitals'])

    # use the __getitem__() method
    print(calc['results']['molecular orbitals'])
    print(calc['results']['geometry optimization history'])

    # perform a normal mode calculation with the optimized structure

    params.update({
        'task': 'normal mode analysis',
        'density convergence': 1.0e-7
    })

    calc = Turbomole(**params)
    mol.calc = calc
    calc.calculate(mol)

    spectrum = calc['results']['vibrational spectrum']
    freq_ref = [0, 0, 0, 0, 0, 0, 1633, 3637, 3745]
    for s in spectrum:
        for freq, mode in zip(freq_ref, range(1, 7)):
            if s['mode number'] == mode:
                assert np.isclose(s['frequency']['value'], freq, rtol=0.05)

    print(calc.todict(skip_default=False))


def test_turbomole_h2o_parsing_issue(turbomole_factory):
    """setting the damping in define causes a parsing issue during restart"""
    mol = molecule('H2O')
    params = {
        'multiplicity': 1,
        'initial damping': 2.0,
        'damping adjustment step': 0.5,
        'minimal damping': 1.0
    }
    mol.calc = Turbomole(**params)
    mol.get_potential_energy(mol)
    Turbomole(restart=True)
