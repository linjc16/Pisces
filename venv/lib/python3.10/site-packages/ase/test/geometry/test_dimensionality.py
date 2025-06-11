import pytest

import ase.build
from ase import Atoms
from ase.geometry.dimensionality import (
    analyze_dimensionality,
    isolate_components,
)
from ase.lattice.cubic import FaceCenteredCubic


@pytest.mark.parametrize("method", ['TSA', 'RDA'])
def test_mx2(method):
    atoms = ase.build.mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19)
    atoms.cell[2, 2] = 7
    atoms.set_pbc((1, 1, 1))
    atoms *= 2

    intervals = analyze_dimensionality(atoms, method=method)
    m = intervals[0]
    assert m.dimtype == '2D'

    assert intervals[0].dimtype == '2D'
    assert intervals[0].h == (0, 0, 2, 0)

    assert intervals[1].dimtype == '3D'
    assert intervals[1].h == (0, 0, 0, 1)

    assert intervals[2].dimtype == '0D'
    assert intervals[2].h == (24, 0, 0, 0)


@pytest.mark.parametrize("method", ['TSA', 'RDA'])
def test_fcc(method):
    atoms = FaceCenteredCubic(size=(2, 2, 2), symbol='Cu', pbc=(1, 1, 1))

    intervals = analyze_dimensionality(atoms, method=method)
    m = intervals[0]
    assert m.dimtype == '3D'


@pytest.mark.parametrize("kcutoff", [None, 1.1])
def test_isolation_0D(kcutoff):
    atoms = ase.build.molecule('H2O', vacuum=3.0)

    result = isolate_components(atoms, kcutoff=kcutoff)
    assert len(result) == 1
    key, components = list(result.items())[0]
    assert key == '0D'
    assert len(components) == 1
    molecule = components[0]
    assert molecule.get_chemical_formula() == atoms.get_chemical_formula()


def test_isolation_1D():
    atoms = Atoms(symbols='Cl6Ti2', pbc=True,
                  cell=[[6.27, 0, 0],
                        [-3.135, 5.43, 0],
                        [0, 0, 5.82]],
                  positions=[[1.97505, 0, 1.455],
                             [0.987525, 1.71044347, 4.365],
                             [-0.987525, 1.71044347, 1.455],
                             [4.29495, 0, 4.365],
                             [2.147475, 3.71953581, 1.455],
                             [-2.147475, 3.71953581, 4.365],
                             [0, 0, 0],
                             [0, 0, 2.91]])

    result = isolate_components(atoms)
    assert len(result) == 1
    key, components = list(result.items())[0]
    assert key == '1D'
    assert len(components) == 1
    chain = components[0]
    assert (chain.pbc == [False, False, True]).all()
    assert chain.get_chemical_formula() == atoms.get_chemical_formula()


def test_isolation_2D():
    atoms = ase.build.mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19)
    atoms.cell[2, 2] = 7
    atoms.set_pbc((1, 1, 1))
    atoms *= 2

    result = isolate_components(atoms)
    assert len(result) == 1
    key, components = list(result.items())[0]
    assert key == '2D'
    assert len(components) == 2
    for layer in components:
        empirical = atoms.get_chemical_formula(empirical=True)
        assert empirical == layer.get_chemical_formula(empirical=True)
        assert (layer.pbc == [True, True, False]).all()


def test_isolation_3D():
    atoms = FaceCenteredCubic(size=(2, 2, 2), symbol='Cu', pbc=(1, 1, 1))

    result = isolate_components(atoms)
    assert len(result) == 1
    key, components = list(result.items())[0]
    assert key == '3D'
    assert len(components) == 1
    bulk = components[0]
    assert bulk.get_chemical_formula() == atoms.get_chemical_formula()
