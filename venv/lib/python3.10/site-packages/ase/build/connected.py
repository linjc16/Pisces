from ase.atoms import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList


def connected_atoms(atoms, index, dmax=None, scale=1.5):
    """Find all atoms connected to atoms[index] and return them."""
    return atoms[connected_indices(atoms, index, dmax, scale)]


def connected_indices(atoms, index, dmax=None, scale=1.5):
    """Find atoms connected to atoms[index] and return their indices.

    If dmax is not None:
    Atoms are defined to be connected if they are nearer than dmax
    to each other.

    If dmax is None:
    Atoms are defined to be connected if they are nearer than the
    sum of their covalent radii * scale to each other.

    """
    if index < 0:
        index = len(atoms) + index

    # set neighbor lists
    if dmax is None:
        # define neighbors according to covalent radii
        radii = scale * covalent_radii[atoms.get_atomic_numbers()]
    else:
        # define neighbors according to distance
        radii = [0.5 * dmax] * len(atoms)
    nl = NeighborList(radii, skin=0, self_interaction=False, bothways=True)
    nl.update(atoms)

    connected = [index] + list(nl.get_neighbors(index)[0])
    isolated = False
    while not isolated:
        isolated = True
        for i in connected:
            for j in nl.get_neighbors(i)[0]:
                if j not in connected:
                    connected.append(j)
                    isolated = False

    return connected


def separate(atoms, **kwargs):
    """Split atoms into separated entities

    Returns:
      List of Atoms object that connected_indices calls connected.
    """
    indices = list(range(len(atoms)))

    separated = []
    while indices:
        my_indcs = connected_indices(atoms, indices[0], **kwargs)
        separated.append(Atoms(cell=atoms.cell, pbc=atoms.pbc))
        for i in my_indcs:
            separated[-1].append(atoms[i])
            del indices[indices.index(i)]

    return separated


def split_bond(atoms, index1, index2, **kwargs):
    """Split atoms by a bond specified by indices

    index1: index of first atom
    index2: index of second atom
    kwargs: kwargs transferred to connected_atoms

    Returns two Atoms objects
    """
    assert index1 != index2
    if index2 > index1:
        shift = 0, 1
    else:
        shift = 1, 0

    atoms_copy = atoms.copy()
    del atoms_copy[index2]
    atoms1 = connected_atoms(atoms_copy, index1 - shift[0], **kwargs)

    atoms_copy = atoms.copy()
    del atoms_copy[index1]
    atoms2 = connected_atoms(atoms_copy, index2 - shift[1], **kwargs)

    return atoms1, atoms2
