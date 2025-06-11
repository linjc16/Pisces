from ase import Atoms
from ase.build import molecule
from ase.build.connected import connected_atoms, separate, split_bond
from ase.data.s22 import data


def test_split_Si2H6():
    """Ensure transferring of arguments from split_bond to connected_atoms"""
    mol = molecule('Si2H6')

    # connect to more atoms using scale
    mol1, mol2 = split_bond(mol, 0, 1, scale=3)
    assert len(mol) < len(mol1) + len(mol2)

    # default scale splits into two SiH3
    mol1, mol2 = split_bond(mol, 0, 1)
    assert len(mol) == len(mol1) + len(mol2)


def test_split_biphenyl():
    mol = molecule('biphenyl')

    mol1, mol2 = split_bond(mol, 0, 14)
    assert len(mol) == len(mol1) + len(mol2)
    mol2s, mol1s = split_bond(mol, 14, 0)
    assert mol1s == mol1
    assert mol2s == mol2

    # we cannot split within the ring
    mol1, mol2 = split_bond(mol, 0, 1)
    assert len(mol) < len(mol1) + len(mol2)


def test_connected_atoms():
    CO = molecule('CO')
    R = CO.get_distance(0, 1)
    assert len(connected_atoms(CO, 0, 1.1 * R)) == 2
    assert len(connected_atoms(CO, 0, 0.9 * R)) == 1

    H2O = molecule('H2O')
    assert len(connected_atoms(H2O, 0)) == 3
    assert len(connected_atoms(H2O, 0, scale=0.9)) == 1

    dimerdata = data['2-pyridoxine_2-aminopyridine_complex']
    dimer = Atoms(dimerdata['symbols'], dimerdata['positions'])

    atoms1 = connected_atoms(dimer, 0)
    atoms2 = connected_atoms(dimer, -1)
    assert len(dimer) == len(atoms1) + len(atoms2)


def test_separate_dimer():
    dimerdata = data['Methanol-formaldehyde_complex']
    dimer = Atoms(dimerdata['symbols'], dimerdata['positions'])

    atoms_list = separate(dimer)
    assert len(atoms_list) == 2
    assert len(atoms_list[0]) + len(atoms_list[1]) == len(dimer)

    # split into atoms
    atoms_list = separate(dimer, scale=1e-5)
    assert len(atoms_list) == len(dimer)
