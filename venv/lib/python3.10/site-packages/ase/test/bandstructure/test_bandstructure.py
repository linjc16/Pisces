import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.test import FreeElectrons
from ase.dft.kpoints import special_paths
from ase.lattice import RHL
from ase.spectrum.band_structure import BandStructure


def test_bandstructure(testdir, plt):
    atoms = bulk('Cu')
    path = special_paths['fcc']
    atoms.calc = FreeElectrons(nvalence=1,
                               kpts={'path': path, 'npoints': 200})
    atoms.get_potential_energy()
    bs = atoms.calc.band_structure()
    _coords, _labelcoords, labels = bs.get_labels()
    print(labels)
    bs.write('hmm.json')
    bs = BandStructure.read('hmm.json')
    _coords, _labelcoords, labels = bs.get_labels()
    print(labels)
    assert ''.join(labels) == 'GXWKGLUWLKUX'
    bs.plot(emax=10, filename='bs.png')
    cols = np.linspace(-1.0, 1.0, bs.energies.size)
    cols.shape = bs.energies.shape
    bs.plot(emax=10, point_colors=cols, filename='bs2.png')


@pytest.fixture()
def bs():
    rhl = RHL(4.0, 65.0)
    path = rhl.bandpath()
    return path.free_electron_band_structure()


def test_print_bs(bs):
    print(bs)


def test_subtract_ref(bs):
    avg = np.mean(bs.energies)
    bs._reference = 5
    bs2 = bs.subtract_reference()
    avg2 = np.mean(bs2.energies)
    assert avg - 5 == pytest.approx(avg2)
