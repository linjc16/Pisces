import numpy as np
import pytest

from ase.build import bulk
from ase.spacegroup import Spacegroup, crystal, utils
from ase.spacegroup.spacegroup import SpacegroupValueError


@pytest.fixture(params=[
    # Use lambda's to not crash during collection if there's an error
    lambda: {
        'atoms': bulk('NaCl', crystalstructure='rocksalt', a=4.1),
        'spacegroup': 225,
        'expected': [[0, 0, 0], [0.5, 0.5, 0.5]]
    },
    # diamond
    lambda: {
        'atoms':
        crystal('C', [(0, 0, 0)],
                spacegroup=227,
                cellpar=[4, 4, 4, 90, 90, 90],
                primitive_cell=True),
        'spacegroup':
        227,
        'expected': [[0, 0, 0]]
    },
    lambda: {
        'atoms':
        crystal('Mg', [(1 / 3, 2 / 3, 3 / 4)],
                spacegroup=194,
                cellpar=[3.21, 3.21, 5.21, 90, 90, 120]),
        'spacegroup':
        194,
        'expected': [(1 / 3, 2 / 3, 3 / 4)]
    },
    lambda: {
        'atoms':
        crystal(['Ti', 'O'],
                basis=[(0, 0, 0), (0.3, 0.3, 0.0)],
                spacegroup=136,
                cellpar=[4, 4, 6, 90, 90, 90]),
        'spacegroup':
        Spacegroup(136),
        'expected': [(0, 0, 0), (0.3, 0.3, 0.0)]
    },
])
def basis_tests(request):
    """Fixture which returns a dictionary with some test inputs and
    expected values for testing the `get_basis` function."""
    return request.param()


def test_get_basis(basis_tests):
    """Test explicitly passing spacegroup and getting basis"""
    atoms = basis_tests['atoms']
    expected = basis_tests['expected']
    spacegroup = basis_tests['spacegroup']

    basis = utils.get_basis(atoms, spacegroup=spacegroup)
    assert np.allclose(basis, expected)


def test_get_basis_infer_sg(basis_tests):
    """Test inferring spacegroup, which uses 'get_basis_spglib' under the
    hood"""
    pytest.importorskip('spglib')

    atoms = basis_tests['atoms']
    expected = basis_tests['expected']

    basis = utils.get_basis(atoms)
    assert np.allclose(basis, expected)


def test_get_basis_spglib(basis_tests):
    """Test getting the basis using spglib"""
    pytest.importorskip('spglib')

    atoms = basis_tests['atoms']
    expected = basis_tests['expected']

    basis = utils._get_basis_spglib(atoms)
    assert np.allclose(basis, expected)


def test_get_basis_ase(basis_tests):
    atoms = basis_tests['atoms']
    spacegroup = basis_tests['spacegroup']
    expected = basis_tests['expected']

    basis = utils._get_basis_ase(atoms, spacegroup)
    assert np.allclose(basis, expected)


@pytest.mark.parametrize('spacegroup', [251.5, [1, 2, 3], np.array([255])])
def test_get_basis_wrong_type(basis_tests, spacegroup):
    atoms = basis_tests['atoms']

    with pytest.raises((SpacegroupValueError, TypeError)):
        utils._get_basis_ase(atoms, spacegroup)
    with pytest.raises((SpacegroupValueError, TypeError)):
        utils.get_basis(atoms, spacegroup=spacegroup)


@pytest.mark.parametrize('method', [None, 12, 'nonsense', True, False])
def test_get_basis_wrong_method(basis_tests, method):
    """Test passing in un-supported methods"""
    atoms = basis_tests['atoms']
    with pytest.raises(ValueError):
        utils.get_basis(atoms, method=method)


def test_get_basis_group_1(basis_tests):
    """Always use spacegroup 1, nothing should be symmetrically equivalent"""
    atoms = basis_tests['atoms']
    scaled = atoms.get_scaled_positions()

    spacegroup = 1

    basis = utils.get_basis(atoms, spacegroup)
    # Basis should now be the same as the scaled positions
    assert np.allclose(basis, scaled)
