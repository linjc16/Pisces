from ase.build import bulk
from ase.calculators.dmol import DMol3


def test_Al_dmol(dmol_factory):
    atoms = bulk('Al')
    calc = DMol3()
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.get_forces()
