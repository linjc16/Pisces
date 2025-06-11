import numpy as np
import pytest

from ase import io
from ase.build import molecule
from ase.io.nwchem.nwreader import _get_multipole


@pytest.fixture()
def atoms():
    return molecule('CH3COOH')


def test_nwchem(atoms):
    """Checks that writing and reading of NWChem input files is consistent."""

    io.write('nwchem.nwi', atoms)
    atoms2 = io.read('nwchem.nwi')

    tol = 1e-8

    check = sum(abs((atoms.positions - atoms2.positions).ravel()) > tol)
    assert check == 0


def test_presteps(atoms):
    """Test the ability to write NWChem input files that perform a series
    of initial guesses for the final wavefunction"""
    test_params = {
        'theory': 'mp2',
        'pretasks': [{'theory': 'dft'}]
    }

    # This should issue a warning if we try to run w/o lindep:n_dep
    with pytest.warns(UserWarning) as record:
        io.write('nwchem.nwi', atoms, **test_params)
    assert 'lindep:n_dep' in str(record[0].message)

    # Make sure the vector output command made it in
    test_params['pretasks'][0]['set'] = {'lindep:n_dep': 0}
    io.write('nwchem.nwi', atoms, **test_params)
    with open('nwchem.nwi') as fp:
        output = fp.read()
    assert "vectors input" in output
    assert "vectors output" in output
    assert "task dft ignore" in output

    # Make a same theory and different basis set
    test_params['basis'] = '6-31g'
    test_params['theory'] = 'dft'
    test_params['pretasks'][0]['basis'] = '3-21g'
    io.write('nwchem.nwi', atoms, **test_params)
    with open('nwchem.nwi') as fp:
        output = fp.read()
    assert "vectors input project smb" in output

    # Add an SCF/3-21g step first
    test_params['pretasks'].insert(
        0, {
            'theory': 'scf',
            'basis': '3-21g',
            'set': {'lindep:n_dep': 0}
        }
    )
    test_params['pretasks'][1].pop('basis')
    io.write('nwchem.nwi', atoms, **test_params)
    with open('nwchem.nwi') as fp:
        output = fp.read()
    assert output.index('task scf ignore') < output.index('task dft ignore')
    assert output.count('3-21g') == 2
    assert output.count('6-31g') == 1
    assert output.index('3-21g') < output.index('6-31g')

    # Test with a charge
    test_params['charge'] = 1
    test_params['scf'] = {'nopen': 1, 'uhf': ''}
    io.write('nwchem.nwi', atoms, **test_params)
    with open('nwchem.nwi') as fp:
        output = fp.read()
    assert output.count('charge 1') == 1
    assert output.index('charge 1') < output.index('task')
    assert output.count('task') == 3


def test_doc_example_1(atoms):
    """Test the first of two examples in the documentation"""
    test_params = {
        'theory': 'mp2',
        'basis': 'aug-cc-pvdz',
        'pretasks': [
            {'dft': {'xc': 'hfexch'},
             'set': {'lindep:n_dep': 0}},
            {'theory': 'scf', 'set': {'lindep:n_dep': 0}}
        ]
    }

    # Make sure there is a dft then SCF
    io.write('nwchem.nwi', atoms, **test_params)

    # Double-check the order of the calculation
    with open('nwchem.nwi') as fp:
        output = fp.read()
    assert output.count('input tmp-') == 2
    assert 'task dft ignore' in output
    assert 'task scf ignore' in output
    assert output.index('dft ignore') < output.index('scf ignore')

    # Double-check that the basis sets are all aug-cc-pvdz
    assert output.count('library aug-cc-pvdz') == 3


def test_doc_example_2(atoms):
    """Test the second example in the documentation"""
    test_params = {
        'theory': 'dft',
        'xc': 'b3lyp',
        'basis': '6-31g(2df,p)',
        'pretasks': [
            {'theory': 'scf', 'basis': '3-21g',
             'set': {'lindep:n_dep': 0}},
            {'dft': {'xc': 'b3lyp'}},
        ]
    }

    # Make sure there is a dft then SCF
    io.write('nwchem.nwi', atoms, **test_params)

    # Double-check the order of the calculation
    with open('nwchem.nwi') as fp:
        output = fp.read()
    assert output.count('input tmp-') == 1  # From 3-21g -> 6-31g
    assert output.count('task dft ignore') == 1
    assert output.count('task scf ignore') == 1
    assert output.index('scf ignore') < output.index('dft ignore')

    # Double-check that the basis sets are first 3-21g then 6-31g
    assert output.count('library 3-21g') == 2
    assert output.count('library 6-31g(2df,p)') == 1


def test_nwchem_trailing_space(datadir):
    """Checks that parsing of NWChem input files works when trailing spaces
    are present in the output file.
    """

    chunk1 = (datadir / 'nwchem/snippet_multipole_7.0.2-gcc.nwo').read_text()
    chunk2 = (datadir / 'nwchem/snippet_multipole_7.0.2-intel.nwo').read_text()

    dipole1, quadrupole1 = _get_multipole(chunk1)
    dipole2, quadrupole2 = _get_multipole(chunk2)

    np.testing.assert_equal(dipole1, dipole2)
    np.testing.assert_equal(quadrupole1, quadrupole2)
