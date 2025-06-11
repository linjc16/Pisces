# fmt: off

import numpy as np

from ase import Atoms
from ase.cluster.util import get_element_info


def Decahedron(symbol, p, q, r, latticeconstant=None):
    """
    Return a decahedral cluster.

    Parameters
    ----------
    symbol: Chemical symbol (or atomic number) of the element.

    p: Number of atoms on the (100) facets perpendicular to the five
    fold axis.

    q: Number of atoms on the (100) facets parallel to the five fold
    axis. q = 1 corresponds to no visible (100) facets.

    r: Depth of the Marks re-entrence at the pentagon corners. r = 0
    corresponds to no re-entrence.

    latticeconstant (optional): The lattice constant. If not given,
    then it is extracted form ase.data.
    """

    symbol, atomic_number, latticeconstant = get_element_info(
        symbol, latticeconstant)

    # Check values of p, q, r
    if p < 1 or q < 1:
        raise ValueError("p and q must be greater than 0.")

    if r < 0:
        raise ValueError("r must be greater than or equal to 0.")

    # Defining constants
    t = 2.0 * np.pi / 5.0
    b = latticeconstant / np.sqrt(2.0)
    a = b * np.sqrt(3.0) / 2.0

    verticies = a * np.array([[np.cos(np.pi / 2.), np.sin(np.pi / 2.), 0.],
                              [np.cos(t * 1. + np.pi / 2.),
                               np.sin(t * 1. + np.pi / 2.), 0.],
                              [np.cos(t * 2. + np.pi / 2.),
                               np.sin(t * 2. + np.pi / 2.), 0.],
                              [np.cos(t * 3. + np.pi / 2.),
                               np.sin(t * 3. + np.pi / 2.), 0.],
                              [np.cos(t * 4. + np.pi / 2.), np.sin(
                                  t * 4. + np.pi / 2.), 0.]])

    # Number of atoms on the five fold axis and a nice constant
    h = p + q + 2 * r - 1
    g = h - q + 1  # p + 2*r

    positions = []
    # Make the five fold axis
    for j in range(h):
        pos = np.array([0.0, 0.0, j * b - (h - 1) * b / 2.0])
        positions.append(pos)

    # Make pentagon rings around the five fold axis
    for n in range(1, h):
        # Condition for (100)-planes
        if n < g:
            for m in range(5):
                v1 = verticies[m - 1]
                v2 = verticies[m]
                for i in range(n):
                    # Condition for marks re-entrence
                    if n - i < g - r and i < g - r:
                        for j in range(h - n):
                            pos = (n - i) * v1 + i * v2
                            pos += np.array([0.0, 0.0, j * b -
                                             (h - n - 1) * b / 2.0])
                            positions.append(pos)

    symbols = [atomic_number] * len(positions)
    return Atoms(symbols=symbols, positions=positions)
