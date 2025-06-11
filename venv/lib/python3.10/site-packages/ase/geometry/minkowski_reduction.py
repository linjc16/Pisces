# fmt: off

import itertools

import numpy as np

from ase.cell import Cell
from ase.utils import pbc2pbc

TOL = 1E-12
MAX_IT = 100000    # in practice this is not exceeded


class CycleChecker:

    def __init__(self, d):
        assert d in [2, 3]

        # worst case is the hexagonal cell in 2D and the fcc cell in 3D
        n = {2: 6, 3: 12}[d]

        # max cycle length is total number of primtive cell descriptions
        max_cycle_length = np.prod([n - i for i in range(d)]) * np.prod(d)
        self.visited = np.zeros((max_cycle_length, 3 * d), dtype=int)

    def add_site(self, H):
        # flatten array for simplicity
        H = H.ravel()

        # check if site exists
        found = (self.visited == H).all(axis=1).any()

        # shift all visited sites down and place current site at the top
        self.visited = np.roll(self.visited, 1, axis=0)
        self.visited[0] = H
        return found


def reduction_gauss(B, hu, hv):
    """Calculate a Gauss-reduced lattice basis (2D reduction)."""
    cycle_checker = CycleChecker(d=2)
    u = hu @ B
    v = hv @ B

    for _ in range(MAX_IT):
        x = int(round(np.dot(u, v) / np.dot(u, u)))
        hu, hv = hv - x * hu, hu
        u = hu @ B
        v = hv @ B
        site = np.array([hu, hv])
        if np.dot(u, u) >= np.dot(v, v) or cycle_checker.add_site(site):
            return hv, hu

    raise RuntimeError(f"Gaussian basis not found after {MAX_IT} iterations")


def relevant_vectors_2D(u, v):
    cs = np.array([e for e in itertools.product([-1, 0, 1], repeat=2)])
    vs = cs @ [u, v]
    indices = np.argsort(np.linalg.norm(vs, axis=1))[:7]
    return vs[indices], cs[indices]


def closest_vector(t0, u, v):
    t = t0
    a = np.zeros(2, dtype=int)
    rs, cs = relevant_vectors_2D(u, v)

    dprev = float("inf")
    for _ in range(MAX_IT):
        ds = np.linalg.norm(rs + t, axis=1)
        index = np.argmin(ds)
        if index == 0 or ds[index] >= dprev:
            return a

        dprev = ds[index]
        r = rs[index]
        kopt = int(round(-np.dot(t, r) / np.dot(r, r)))
        a += kopt * cs[index]
        t = t0 + a[0] * u + a[1] * v

    raise RuntimeError(f"Closest vector not found after {MAX_IT} iterations")


def reduction_full(B):
    """Calculate a Minkowski-reduced lattice basis (3D reduction)."""
    cycle_checker = CycleChecker(d=3)
    H = np.eye(3, dtype=int)
    norms = np.linalg.norm(B, axis=1)

    for _ in range(MAX_IT):
        # Sort vectors by norm
        H = H[np.argsort(norms, kind='merge')]

        # Gauss-reduce smallest two vectors
        hw = H[2]
        hu, hv = reduction_gauss(B, H[0], H[1])
        H = np.array([hu, hv, hw])
        R = H @ B

        # Orthogonalize vectors using Gram-Schmidt
        u, v, _ = R
        X = u / np.linalg.norm(u)
        Y = v - X * np.dot(v, X)
        Y /= np.linalg.norm(Y)

        # Find closest vector to last element of R
        pu, pv, pw = R @ np.array([X, Y]).T
        nb = closest_vector(pw, pu, pv)

        # Update basis
        H[2] = [nb[0], nb[1], 1] @ H
        R = H @ B

        norms = np.linalg.norm(R, axis=1)
        if norms[2] >= norms[1] or cycle_checker.add_site(H):
            return R, H

    raise RuntimeError(f"Reduced basis not found after {MAX_IT} iterations")


def is_minkowski_reduced(cell, pbc=True):
    """Tests if a cell is Minkowski-reduced.

    Parameters:

    cell: array
        The lattice basis to test (in row-vector format).
    pbc: array, optional
        The periodic boundary conditions of the cell (Default `True`).
        If `pbc` is provided, only periodic cell vectors are tested.

    Returns:

    is_reduced: bool
        True if cell is Minkowski-reduced, False otherwise.
    """

    """These conditions are due to Minkowski, but a nice description in English
    can be found in the thesis of Carine Jaber: "Algorithmic approaches to
    Siegel's fundamental domain", https://www.theses.fr/2017UBFCK006.pdf
    This is also good background reading for Minkowski reduction.

    0D and 1D cells are trivially reduced. For 2D cells, the conditions which
    an already-reduced basis fulfil are:
    |b1| ≤ |b2|
    |b2| ≤ |b1 - b2|
    |b2| ≤ |b1 + b2|

    For 3D cells, the conditions which an already-reduced basis fulfil are:
    |b1| ≤ |b2| ≤ |b3|

    |b1 + b2|      ≥ |b2|
    |b1 + b3|      ≥ |b3|
    |b2 + b3|      ≥ |b3|
    |b1 - b2|      ≥ |b2|
    |b1 - b3|      ≥ |b3|
    |b2 - b3|      ≥ |b3|
    |b1 + b2 + b3| ≥ |b3|
    |b1 - b2 + b3| ≥ |b3|
    |b1 + b2 - b3| ≥ |b3|
    |b1 - b2 - b3| ≥ |b3|
    """
    pbc = pbc2pbc(pbc)
    dim = pbc.sum()
    if dim <= 1:
        return True

    if dim == 2:
        # reorder cell vectors to [shortest, longest, aperiodic]
        cell = cell.copy()
        cell[np.argmin(pbc)] = 0
        norms = np.linalg.norm(cell, axis=1)
        cell = cell[np.argsort(norms)[[1, 2, 0]]]

        A = [[0, 1, 0],
             [1, -1, 0],
             [1, 1, 0]]
        lhs = np.linalg.norm(A @ cell, axis=1)
        norms = np.linalg.norm(cell, axis=1)
        rhs = norms[[0, 1, 1]]
    else:
        A = [[0, 1, 0],
             [0, 0, 1],
             [1, 1, 0],
             [1, 0, 1],
             [0, 1, 1],
             [1, -1, 0],
             [1, 0, -1],
             [0, 1, -1],
             [1, 1, 1],
             [1, -1, 1],
             [1, 1, -1],
             [1, -1, -1]]
        lhs = np.linalg.norm(A @ cell, axis=1)
        norms = np.linalg.norm(cell, axis=1)
        rhs = norms[[0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2]]
    return (lhs >= rhs - TOL).all()


def minkowski_reduce(cell, pbc=True):
    """Calculate a Minkowski-reduced lattice basis.  The reduced basis
    has the shortest possible vector lengths and has
    norm(a) <= norm(b) <= norm(c).

    Implements the method described in:

    Low-dimensional Lattice Basis Reduction Revisited
    Nguyen, Phong Q. and Stehlé, Damien,
    ACM Trans. Algorithms 5(4) 46:1--46:48, 2009
    :doi:`10.1145/1597036.1597050`

    Parameters:

    cell: array
        The lattice basis to reduce (in row-vector format).
    pbc: array, optional
        The periodic boundary conditions of the cell (Default `True`).
        If `pbc` is provided, only periodic cell vectors are reduced.

    Returns:

    rcell: array
        The reduced lattice basis.
    op: array
        The unimodular matrix transformation (rcell = op @ cell).
    """
    cell = Cell(cell)
    pbc = pbc2pbc(pbc)
    dim = pbc.sum()
    op = np.eye(3, dtype=int)
    if is_minkowski_reduced(cell, pbc):
        return cell, op

    if dim == 2:
        # permute cell so that first two vectors are the periodic ones
        perm = np.argsort(pbc, kind='merge')[::-1]    # stable sort
        pcell = cell[perm][:, perm]

        # perform gauss reduction
        norms = np.linalg.norm(pcell, axis=1)
        norms[2] = float("inf")
        indices = np.argsort(norms)
        op = op[indices]
        hu, hv = reduction_gauss(pcell, op[0], op[1])
        op[0] = hu
        op[1] = hv

        # undo above permutation
        invperm = np.argsort(perm)
        op = op[invperm][:, invperm]

        # maintain cell handedness
        index = np.argmin(pbc)
        normal = np.cross(cell[index - 2], cell[index - 1])
        normal /= np.linalg.norm(normal)

        _cell = cell.copy()
        _cell[index] = normal
        _rcell = op @ cell
        _rcell[index] = normal
        if _cell.handedness != Cell(_rcell).handedness:
            op[index - 1] *= -1

    elif dim == 3:
        _, op = reduction_full(cell)
        # maintain cell handedness
        if cell.handedness != Cell(op @ cell).handedness:
            op = -op

    norms1 = np.sort(np.linalg.norm(cell, axis=1))
    norms2 = np.sort(np.linalg.norm(op @ cell, axis=1))
    if (norms2 > norms1 + TOL).any():
        raise RuntimeError("Minkowski reduction failed")
    return op @ cell, op
