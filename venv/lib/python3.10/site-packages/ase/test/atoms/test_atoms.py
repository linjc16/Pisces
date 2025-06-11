import numpy as np
import pytest

from ase import Atoms


def array_almost_equal(a1, a2, tol=np.finfo(float).eps):
    return (np.abs(a1 - a2) < tol).all()


def test_atoms():
    print(Atoms())
    print(Atoms('H2O'))
    # ...


def test_numbers_input():
    numbers = np.array([[0, 1], [2, 3]])
    with pytest.raises(Exception, match='"numbers" must be 1-dimensional.'):
        Atoms(positions=np.zeros((2, 3)), numbers=numbers, cell=np.eye(3))

    Atoms(positions=np.zeros((2, 3)), numbers=[0, 1], cell=np.eye(3))


def test_bad_array_shape():
    with pytest.raises(ValueError, match='wrong length'):
        Atoms().set_masses([1, 2])

    with pytest.raises(ValueError, match='wrong length'):
        Atoms('H').set_masses([])

    with pytest.raises(ValueError, match='wrong shape'):
        Atoms('H').set_masses(np.ones((1, 3)))


def test_set_masses():
    atoms = Atoms('AgAu')
    m0 = atoms.get_masses()
    atoms.set_masses([1, None])
    assert atoms.get_masses() == pytest.approx([1, m0[1]])


def test_ndof():
    a = Atoms('CO')
    b = Atoms('H2O')

    assert a.get_number_of_degrees_of_freedom() == 6
    assert b.get_number_of_degrees_of_freedom() == 9

    c = a + b

    assert c.get_number_of_degrees_of_freedom() == 15

    del c[0]

    assert c.get_number_of_degrees_of_freedom() == 12


@pytest.mark.parametrize('zlength', [0, 10])
def test_get_com(zlength):
    """Test that atoms.get_center_of_mass(scaled=True) works"""

    d = 1.142
    a = Atoms('CO', positions=[(2, 0, 0), (2, -d, 0)], pbc=True)
    a.set_cell(np.array(((4, -4, 0), (0, 5.657, 0), (0, 0, zlength))))

    scaledref = np.array((0.5, 0.23823622, 0.))
    assert array_almost_equal(a.get_center_of_mass(scaled=True),
                              scaledref, tol=1e-8)


@pytest.mark.parametrize('indices', [[0, 1], slice(0, 2), "0:2"])
def test_get_com_indices(indices):
    """Test get_center_of_mass(indices=...)"""
    positions = [
        [-0.3, 0.0, +0.3],
        [-0.3, 0.0, -0.3],
        [+0.3, 0.0, +0.3],
        [+0.3, 0.0, -0.3],
    ]
    atoms = Atoms('HHHH', positions=positions)
    np.testing.assert_almost_equal(
        atoms.get_center_of_mass(indices=indices),
        [-0.3, 0.0, 0.0],
    )


@pytest.mark.parametrize('scaled', [True, False])
@pytest.mark.parametrize('symbols', ['CO', 'H2'])
def test_set_com(scaled, symbols):
    """Test that atoms.set_center_of_mass() works. """
    a = Atoms(symbols, positions=[(0, 0, 0), (0, 0, 1)],
              cell=[2, 2, 2], pbc=True)

    desired_com = [0.1, 0.5, 0.6]
    a.set_center_of_mass(desired_com, scaled=scaled)

    assert array_almost_equal(a.get_center_of_mass(scaled=scaled), desired_com)
