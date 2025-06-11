"""Tests for the NWChem computations which use more than one task"""
import numpy as np
import pytest

import ase
from ase.build import molecule


@pytest.fixture()
def atoms() -> ase.Atoms:
    return molecule('H2O')


@pytest.mark.calculator('nwchem')
@pytest.mark.parametrize(
    'params',
    [{  # Documentation example
        'theory': 'mp2',
        'basis': 'aug-cc-pvdz',
        'pretasks': [
            {'dft': {'xc': 'hfexch'},
             'set': {'lindep:n_dep': 0}},
            {'theory': 'scf', 'set': {'lindep:n_dep': 0}}
        ]
    }, {  # Increase basis set, then theory
        'theory': 'mp2',
        'basis': 'aug-cc-pvdz',
        'pretasks': [
            {'dft': {'xc': 'hfexch'}, 'basis': '3-21g',
             'set': {'lindep:n_dep': 0}},
            {'dft': {'xc': 'hfexch'}, 'basis': 'aug-cc-pvdz',
             'set': {'lindep:n_dep': 0}},
            {'theory': 'scf', 'set': {'lindep:n_dep': 0}}
        ]
    }, {  # Charged
        'theory': 'mp2',
        'basis': 'aug-cc-pvdz',
        'charge': 1,
        'scf': {'uhf': ''},
        'pretasks': [
            {'dft': {'xc': 'hfexch'},
             'set': {'lindep:n_dep': 0}},
            {'theory': 'scf',
             'set': {'lindep:n_dep': 0}}
        ]
    }]
)
def test_example(factory, atoms, params):
    """Make sure the example in the documentation works"""

    # If there is a charge, update the atoms object such that
    #  it has a nonzero total magnetic moment
    #  This will test the "_update_mult" feature of nwchem writer
    if 'charge' in params:
        magmoms = atoms.get_initial_magnetic_moments()
        magmoms[0] = params['charge']
        atoms.set_initial_magnetic_moments(magmoms)

    # Get the energy with the pretasks
    calc = factory.calc(**params)
    with_eng = calc.get_potential_energy(atoms)

    # Make sure the vectors loaded in correctly
    with open(f'{calc.label}.nwi') as fp:
        input_file = fp.read()
    assert input_file.count('task') == len(params['pretasks']) + 1
    with open(f'{calc.label}.nwo') as fp:
        output = fp.read()
    assert (output.count('Loading old vectors from job with title')
            + output.count('Orbital projection guess')) \
        == len(params['pretasks'])
    assert 'Load of old vectors failed' not in output, input_file

    # Get it without
    params.pop('pretasks')
    calc = factory.calc(**params)
    without_eng = calc.get_potential_energy(atoms)

    assert np.isclose(without_eng, with_eng)
