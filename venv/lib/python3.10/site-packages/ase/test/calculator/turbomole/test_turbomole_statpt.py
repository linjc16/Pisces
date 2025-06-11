# type: ignore
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ase import Atoms
from ase.calculators.turbomole import Turbomole
from ase.io import jsonio


def test_turbomole_statpt(turbomole_factory):
    """test transition state optimization using the statpt module"""
    json_atoms = ('{"numbers": [6, 6, 6, 8, 1, 1, 1, 1], "positions": [[0.0, 0'
                  '.0, 0.0], [0.0, 0.0, 1.3399984079060756], [1.25573752066799'
                  '53, 0.0, 2.0650029370111715], [1.7840151299738778, 1.056549'
                  '9268396596, 2.3699995140917474], [-0.9353048443552724, 0.0,'
                  ' -0.5399988845198718], [0.9353048443552724, 0.0, -0.5399988'
                  '845198718], [-0.9353048443552724, 0.0, 1.8800025841980528],'
                  ' [1.723392588731684, -0.9353048443552724, 2.334999733385054'
                  '7]]}')

    dist_mat_ref = [1.30849669, 2.50120360, 3.30620272, 1.08211169, 1.08253121,
                    2.08500900, 3.07726929, 1.52724105, 2.43138647, 2.09303454,
                    2.09733683, 1.08519591, 2.22821075, 1.21863941, 3.49692549,
                    2.75562078, 2.22228794, 1.10557792, 4.28911312, 3.42667536,
                    3.01432199, 2.02448325, 1.83319672, 2.43773514, 4.05344281,
                    3.06311635, 3.21465341, 2.83929402]

    atoms = Atoms(**jsonio.decode(json_atoms))
    calc = Turbomole(**{'task': 'optimize', 'transition vector': 1,
                        'use dft': False, 'basis set name': 'sto-3g hondo',
                        'multiplicity': 1})
    atoms.calc = calc
    calc.calculate()
    assert calc.converged
    assert atoms is not calc.atoms
    dist_mat = calc.atoms.get_all_distances()[np.triu_indices(len(atoms), k=1)]
    assert_allclose(dist_mat, dist_mat_ref)

    calc = Turbomole(task='frequencies', restart=True)
    calc.calculate()
    assert calc.converged
    spectrum = calc.results['vibrational spectrum']
    assert spectrum[0]['frequency']['units'] == 'cm^-1'
    assert spectrum[0]['frequency']['value'] == pytest.approx(-164.78)
    assert spectrum[0]['irreducible representation'] == 'a'
