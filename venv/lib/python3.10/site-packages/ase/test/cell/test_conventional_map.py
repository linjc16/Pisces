import pytest

from ase import Atoms
from ase.build.supercells import make_supercell
from ase.calculators.emt import EMT
from ase.lattice import all_variants


def emt_energy_per_atom(atoms):
    atoms = atoms.copy()
    atoms.calc = EMT()
    return atoms.get_potential_energy() / len(atoms)


@pytest.mark.parametrize('lat', [var for var in all_variants()
                                 if var.ndim == 3])
def test_conventional_map(lat):
    if not hasattr(lat, 'conventional_cellmap'):
        pytest.skip()

    conv_lat = lat.conventional()
    prim_atoms = Atoms('Au', cell=lat.tocell(), pbc=1)
    conv_atoms = make_supercell(prim_atoms, lat.conventional_cellmap)

    e1 = emt_energy_per_atom(prim_atoms)
    e2 = emt_energy_per_atom(conv_atoms)

    assert e1 == pytest.approx(e2)
    assert conv_lat.cellpar() == pytest.approx(conv_atoms.cell.cellpar())

    # Rule out also that cells could differ by a rotation:
    assert conv_lat.tocell()[:] == pytest.approx(conv_atoms.cell[:])
