import pytest

from ase import Atoms
from ase.calculators.test import FreeElectrons
from ase.lattice import all_variants
from ase.spectrum.band_structure import calculate_band_structure


@pytest.mark.parametrize("i, lat",
                         [pytest.param(i, lat, id=lat.variant)
                          for i, lat in enumerate(all_variants())
                          if lat.ndim == 3])
def test_lattice_bandstructure(testdir, i, lat, figure):
    xid = f'{i:02d}.{lat.variant}'
    path = lat.bandpath(density=10)
    path.write(f'path.{xid}.json')
    atoms = Atoms(cell=lat.tocell(), pbc=True)
    atoms.calc = FreeElectrons(nvalence=0, kpts=path.kpts)
    bs = calculate_band_structure(atoms, path)
    bs.write(f'bs.{xid}.json')

    ax = figure.gca()
    bs.plot(ax=ax, emin=0, emax=20, filename=f'fig.{xid}.png')
