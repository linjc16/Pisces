import numpy as np

from ase.data import covalent_radii
from ase.neighborlist import NeighborList


def get_bond_list(atoms, nl, rs):
    bonds = []
    for i in range(len(atoms)):
        p = atoms.positions[i]
        indices, offsets = nl.get_neighbors(i)

        for j, offset in zip(indices, offsets):
            q = atoms.positions[j] + np.dot(offset, atoms.get_cell())
            d = np.linalg.norm(p - q)
            k = d / (rs[i] + rs[j])
            bonds.append((k, i, j, tuple(offset)))
    return set(bonds)


def next_bond(atoms):
    """
    Generates bonds (lazily) one at a time, sorted by k-value (low to high).
    Here, k = d_ij / (r_i + r_j), where d_ij is the bond length and r_i and r_j
    are the covalent radii of atoms i and j.

    Parameters:

    atoms: ASE atoms object

    Returns: iterator of bonds
        A bond is a tuple with the following elements:

        k:       float   k-value
        i:       float   index of first atom
        j:       float   index of second atom
        offset:  tuple   cell offset of second atom
    """
    kmax = 0
    rs = covalent_radii[atoms.get_atomic_numbers()]
    seen = set()
    while 1:
        # Expand the scope of the neighbor list.
        kmax += 2
        nl = NeighborList(kmax * rs, skin=0, self_interaction=False)
        nl.update(atoms)

        # Get a list of bonds, sorted by k-value.
        bonds = get_bond_list(atoms, nl, rs)
        new_bonds = bonds - seen
        if len(new_bonds) == 0:
            break

        # Yield the bonds which we have not previously generated.
        seen.update(new_bonds)
        yield from sorted(new_bonds, key=lambda x: x[0])
