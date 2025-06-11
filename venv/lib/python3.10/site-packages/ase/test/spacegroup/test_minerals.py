from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from ase.io import read
from ase.spacegroup.symmetrize import (
    IntermediateDatasetError,
    check_symmetry,
    get_symmetrized_atoms,
)

spglib = pytest.importorskip('spglib')


@dataclass
class Mineral:
    name: str
    spacegroup: int
    cod_id: str  # Crystallographic Open Database entry

    @property
    def datafile(self):
        return Path(f'minerals/cod_{self.cod_id}.cif')


# Named minerals in testdata
named_minerals = [
    # Centrosymmetric triclinic
    Mineral(name='artroeite', spacegroup=2, cod_id='9001665'),
    # Enantiomorphic (chiral) polar monoclinic
    Mineral(name='alloclasite', spacegroup=4, cod_id='9004112'),
    # Polar orthorhombic
    Mineral(name='cobaltite', spacegroup=29, cod_id='9004218'),
    # Enantiomorphic (chiral) tetragonal
    Mineral(name='cristobalite', spacegroup=92, cod_id='9017338'),
    # Enantiomorphic (chiral) trigonal
    Mineral(name='heazlewoodite', spacegroup=155, cod_id='9007640'),
    # Polar trigonal
    Mineral(name='molybdenite 3R', spacegroup=160, cod_id='9007661'),
    # Centrosymmetric hexagonal
    Mineral(name='breithauptite', spacegroup=194, cod_id='1010930'),
    # Cubic
    Mineral(name='moissanite 3C', spacegroup=216, cod_id='1010995'),
]


@pytest.mark.parametrize('mineral', list(named_minerals))
def test_mineral_spacegroups(datadir, mineral):
    atoms = read(datadir / mineral.datafile)
    dataset = check_symmetry(atoms)
    assert dataset.number == mineral.spacegroup


@pytest.mark.parametrize('mineral,rngseed', product(
    list(named_minerals), [13, 42, 93]))
def test_mineral_symmetrization(datadir, mineral, rngseed):
    atoms = read(datadir / mineral.datafile)
    assert mineral.spacegroup > 1  # some symmetry

    # Break the symmetry
    rng = np.random.default_rng(rngseed)
    cell = atoms.get_cell()
    atoms.set_cell(cell.array + rng.normal(scale=0.01, size=(3, 3)))
    atoms.rattle(0.01, rng=rng)
    rattled_dataset = check_symmetry(atoms)
    assert rattled_dataset.number == 1

    # Find a symmetry precision that recovers the original symmetry
    symprec = 1e-5
    _symatoms, dataset = get_symmetrized_atoms(atoms, symprec=symprec)
    while dataset.number != mineral.spacegroup:
        if symprec > 0.5:
            raise ValueError('Could not recover original symmetry of the'
                             f'mineral {mineral.name}')
        symprec *= 1.2
        try:
            _symatoms, dataset = get_symmetrized_atoms(
                atoms, symprec=symprec, final_symprec=1e-5)
        except IntermediateDatasetError:
            continue
