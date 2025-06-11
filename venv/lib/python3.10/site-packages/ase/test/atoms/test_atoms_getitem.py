import pytest

from ase.atoms import Atoms


def test_h2o_getitem():
    w = Atoms('H2O',
              positions=[[2.264, 0.639, 0.876],
                         [0.792, 0.955, 0.608],
                         [1.347, 0.487, 1.234]],
              cell=[3, 3, 3],
              pbc=True)

    with pytest.raises(IndexError):
        w[True, False]

    assert w[0, 1] == w[True, True, False]
    assert w[0, 1] == w[0:2]


@pytest.mark.parametrize(
    'symbols, index, expected',
    [
        ('', [], ''),
        ('', slice(None), ''),
        ('X', [], ''),
        ('XY', slice(4, 3), ''),
        ('XY', slice(None), 'XY'),
        ('HHeLiBe', 1, 'He'),
        ('HHeLiBe', -1, 'Be'),
        ('HHeLiBe', -2, 'Li'),
        ('HHeLiBe', slice(1, 3), 'HeLi'),
        ('HHeLiBe', slice(1, -1), 'HeLi'),
        ('HHeLiBe', [True, False, False, True], 'HBe'),
        ('HHeLiBeBCNOF', slice(1, 7, 2), 'HeBeC'),
    ])
def test_getitem(symbols, index, expected):
    """Test various slicing syntaxes on various simple atoms objects."""
    atoms = Atoms(symbols)
    indexed_atoms = atoms[index]

    expected = str(Atoms(expected).symbols)

    if isinstance(indexed_atoms, Atoms):
        assert str(indexed_atoms.symbols) == expected
    else:
        # Single atom
        assert indexed_atoms.symbol == expected
