# type: ignore
import numpy as np

from ase.calculators.turbomole import Turbomole
from ase.cluster.cubic import FaceCenteredCubic


def test_turbomole_au13(turbomole_factory):
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    layers = [1, 2, 1]
    atoms = FaceCenteredCubic('Au', surfaces, layers, latticeconstant=4.08)

    params = {
        'title': 'Au13-',
        'task': 'energy',
        'basis set name': 'def2-SV(P)',
        'total charge': -1,
        'multiplicity': 1,
        'use dft': True,
        'density functional': 'pbe',
        'use resolution of identity': True,
        'ri memory': 1000,
        'use fermi smearing': True,
        'fermi initial temperature': 500.,
        'fermi final temperature': 100.,
        'fermi annealing factor': 0.9,
        'fermi homo-lumo gap criterion': 0.09,
        'fermi stopping criterion': 0.002,
        'scf energy convergence': 1.e-4,
        'scf iterations': 250
    }

    calc = Turbomole(**params)
    atoms.calc = calc
    calc.calculate(atoms)

    # use the get_property() method
    assert np.isclose(calc.get_property('energy'), -48044.567169, atol=1e-4)
    dipole = calc.get_property('dipole')
    dipole_ref = [1.68659890e-09, 1.17584764e-09, -1.45238506e-09]
    assert np.allclose(dipole, dipole_ref, rtol=0.01)

    # test restart

    params = {
        'task': 'gradient',
        'scf energy convergence': 1.e-6
    }

    calc = Turbomole(restart=True, **params)
    assert calc.converged
    calc.calculate()

    assert np.isclose(calc.get_property('energy'), -48044.567179, atol=1e-5)
    force = np.linalg.norm(calc.get_property('forces'))
    force_ref = 0.27110367946343794
    assert np.isclose(force, force_ref, rtol=0.01)
    dipole = calc.get_property('dipole')
    dipole_ref = [5.97945377e-09, 2.72637920e-09, -3.68399945e-09]
    assert np.allclose(dipole, dipole_ref, rtol=0.01)
