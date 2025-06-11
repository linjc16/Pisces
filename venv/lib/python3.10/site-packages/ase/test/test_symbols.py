import pytest

from ase import Atoms
from ase.build import molecule
from ase.symbols import Symbols


@pytest.fixture()
def atoms():
    return molecule('CH3CH2OH')


@pytest.fixture()
def symbols(atoms):
    return atoms.symbols


def test_symbols_indexing(atoms):
    print(atoms.symbols)
    atoms.symbols[0] = 'X'
    atoms.symbols[2:4] = 'Pu'
    atoms.numbers[6:8] = 79

    assert atoms.numbers[0] == 0
    assert (atoms.numbers[2:4] == 94).all()
    assert sum(atoms.symbols == 'Au') == 2
    assert (atoms.symbols[6:8] == 'Au').all()
    assert (atoms.symbols[:3] == 'XCPu').all()

    print(atoms)
    print(atoms.numbers)


def test_symbols_vs_get_chemical_symbols(atoms):
    assert atoms.get_chemical_symbols() == list(atoms.symbols)


def test_str_roundtrip(symbols):
    string = str(symbols)
    newsymbols = Symbols.fromsymbols(string)
    assert (symbols == newsymbols).all()


def test_manipulation_with_string():
    atoms = molecule('H2O')
    atoms.symbols = 'Au2Ag'
    print(atoms.symbols)
    assert (atoms.symbols == 'Au2Ag').all()


def test_search(atoms):
    indices = atoms.symbols.search('H')
    assert len(indices) > 0
    assert (atoms.symbols[indices] == 'H').all()
    assert (atoms[indices].symbols == 'H').all()


def test_search_two(atoms):
    indices = atoms.symbols.search('CO')
    assert all(sym in {'C', 'O'} for sym in atoms.symbols[indices])


def test_species(atoms):
    assert atoms.symbols.species() == set(atoms.symbols)


def test_indices(atoms):
    dct = atoms.symbols.indices()

    assert set(dct) == atoms.symbols.species()

    for symbol, indices in dct.items():
        assert all(atoms.symbols[indices] == symbol)


def test_symbols_to_symbols(symbols):
    assert all(Symbols(symbols.numbers) == symbols)


def test_symbols_to_atoms(symbols):
    assert all(Atoms(symbols).symbols == symbols)


def test_symbols_to_formula():
    symstr = 'CH3CH2OH'
    symbols = Symbols.fromsymbols(symstr)
    assert str(symbols.formula) == symstr
