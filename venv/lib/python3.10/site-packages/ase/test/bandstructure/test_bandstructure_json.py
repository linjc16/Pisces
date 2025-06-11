from ase.build import bulk
from ase.calculators.test import FreeElectrons
from ase.io.jsonio import read_json
from ase.spectrum.band_structure import BandStructure, calculate_band_structure


def test_bandstructure_json(testdir):
    atoms = bulk('Au')
    lat = atoms.cell.get_bravais_lattice()
    path = lat.bandpath(npoints=100)

    atoms.calc = FreeElectrons()

    bs = calculate_band_structure(atoms, path)
    bs.write('bs.json')
    bs.path.write('path.json')

    bs1 = read_json('bs.json')
    bs2 = BandStructure.read('bs.json')
    path1 = read_json('path.json')
    assert type(bs1) == type(bs)  # noqa
    assert type(bs2) == type(bs)  # noqa
    assert type(path1) == type(bs.path)  # noqa
