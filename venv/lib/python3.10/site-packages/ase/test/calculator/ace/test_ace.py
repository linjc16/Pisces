from ase import Atoms
from ase.calculators.acemolecule import ACE


def test_ace(ace_factory):
    label = "test"
    mol = Atoms('H2', [(0, 0, 0), (0, 0, 0.7)])
    basic = [dict(Cell='5.0')]
    ace = ACE(label=label, BasicInformation=basic)
    mol.calc = ace
    forces = mol.get_forces()
    print(forces)
