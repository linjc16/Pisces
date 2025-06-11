import pytest

from ase.build import bulk
from ase.spectrum.band_structure import calculate_band_structure


@pytest.mark.calculator_lite()
@pytest.mark.calculator('siesta')
def test_bands(factory):
    atoms = bulk('Si')
    path = atoms.cell.bandpath('GXWK', density=10)
    atoms.calc = factory.calc(kpts=[2, 2, 2])
    bs = calculate_band_structure(atoms, path)
    print(bs)
    bs.write('bs.json')
