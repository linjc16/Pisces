import numpy as np

from ase import Atoms
from ase.build import bulk, make_supercell


def test_bulk():
    a1 = bulk('ZnS', 'wurtzite', a=3.0, u=0.23) * (1, 2, 1)
    a2 = bulk('ZnS', 'wurtzite', a=3.0, u=0.23, orthorhombic=True)
    a1.cell = a2.cell
    a1.wrap()
    assert abs(a1.positions - a2.positions).max() < 1e-14


class TestCubic:
    def test_sc(self):
        name = "Po"
        structure = "sc"
        a = 1.0
        atoms0 = bulk(name, structure, a=a)
        atoms1 = bulk(name, structure, a=a, cubic=True)
        self.compare(atoms0, atoms1)

    def test_bcc(self):
        name = "Li"
        structure = "bcc"
        a = 1.0
        P = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        atoms0 = make_supercell(bulk(name, structure, a=a), P)
        atoms1 = bulk(name, structure, a=a, cubic=True)
        self.compare(atoms0, atoms1)

    def test_cesiumchloride(self):
        name = "CsCl"
        structure = "cesiumchloride"
        a = 1.0
        atoms0 = bulk(name, structure, a=a)
        atoms1 = bulk(name, structure, a=a, cubic=True)
        self.compare(atoms0, atoms1)

    def test_fcc(self):
        name = "Cu"
        structure = "fcc"
        a = 1.0
        P = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        atoms0 = make_supercell(bulk(name, structure, a=a), P)
        atoms0 = atoms0[[0, 3, 2, 1]]  # sort
        atoms1 = bulk(name, structure, a=a, cubic=True)
        self.compare(atoms0, atoms1)

    def test_diamond(self):
        name = "C"
        structure = "diamond"
        a = 1.0
        P = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        atoms0 = make_supercell(bulk(name, structure, a=a), P)
        atoms0 = atoms0[[0, 1, 6, 7, 4, 5, 2, 3]]  # sort
        atoms1 = bulk(name, structure, a=a, cubic=True)
        self.compare(atoms0, atoms1)

    def test_zincblende(self):
        name = "ZnS"
        structure = "zincblende"
        a = 1.0
        P = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        atoms0 = make_supercell(bulk(name, structure, a=a), P)
        atoms0 = atoms0[[0, 1, 6, 7, 4, 5, 2, 3]]  # sort
        atoms1 = bulk(name, structure, a=a, cubic=True)
        self.compare(atoms0, atoms1)

    def test_rocksalt(self):
        name = "NaCl"
        structure = "rocksalt"
        a = 1.0
        P = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        atoms0 = make_supercell(bulk(name, structure, a=a), P)
        atoms0 = Atoms(atoms0[[0, 1, 6, 7, 4, 5, 2, 3]])  # sort
        atoms1 = bulk(name, structure, a=a, cubic=True)
        self.compare(atoms0, atoms1)

    def test_fluorite(self):
        name = "CaF2"
        structure = "fluorite"
        a = 1.0
        P = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        atoms0 = make_supercell(bulk(name, structure, a=a), P)
        atoms0 = atoms0[[0, 1, 2, 9, 10, 11, 6, 7, 8, 3, 4, 5]]  # sort
        atoms1 = bulk(name, structure, a=a, cubic=True)
        self.compare(atoms0, atoms1)

    def compare(self, atoms0: Atoms, atoms1: Atoms):
        a = 1e-14
        print(atoms0.positions)
        print(atoms1.positions)
        np.testing.assert_allclose(atoms0.cell, atoms1.cell, atol=a)
        np.testing.assert_allclose(atoms0.positions, atoms1.positions, atol=a)


def hasmom(*args, **kwargs):
    return bulk(*args, **kwargs).has('initial_magmoms')


def test_magnetic_or_not():
    assert hasmom('Fe')
    assert hasmom('Fe', orthorhombic=True)
    assert hasmom('Fe', cubic=True)
    assert hasmom('Fe', 'bcc', 4.0)
    assert not hasmom('Fe', 'fcc', 4.0)
    assert not hasmom('Ti')
    assert not hasmom('Ti', 'bcc', 4.0)

    assert hasmom('Co')
    assert hasmom('Ni')
