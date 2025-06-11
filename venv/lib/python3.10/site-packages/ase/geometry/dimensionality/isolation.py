# fmt: off

"""
Implements functions for extracting ('isolating') a low-dimensional material
component in its own unit cell.

This uses the rank-determination method described in:
Definition of a scoring parameter to identify low-dimensional materials
components
P.M. Larsen, M. Pandey, M. Strange, and K. W. Jacobsen
Phys. Rev. Materials 3 034003, 2019
https://doi.org/10.1103/PhysRevMaterials.3.034003
"""
import collections
import itertools

import numpy as np

from ase import Atoms
from ase.geometry.cell import complete_cell
from ase.geometry.dimensionality import (
    analyze_dimensionality,
    rank_determination,
)
from ase.geometry.dimensionality.bond_generator import next_bond
from ase.geometry.dimensionality.interval_analysis import merge_intervals


def orthogonal_basis(X, Y=None):

    is_1d = Y is None
    b = np.zeros((3, 3))
    b[0] = X
    if not is_1d:
        b[1] = Y
    b = complete_cell(b)

    Q = np.linalg.qr(b.T)[0].T
    if np.dot(b[0], Q[0]) < 0:
        Q[0] = -Q[0]

    if np.dot(b[2], Q[1]) < 0:
        Q[1] = -Q[1]

    if np.linalg.det(Q) < 0:
        Q[2] = -Q[2]

    if is_1d:
        Q = Q[[1, 2, 0]]

    return Q


def select_cutoff(atoms):
    intervals = analyze_dimensionality(atoms, method='RDA', merge=False)
    dimtype = max(merge_intervals(intervals), key=lambda x: x.score).dimtype
    m = next(e for e in intervals if e.dimtype == dimtype)
    if m.b == float("inf"):
        return m.a + 0.1
    else:
        return (m.a + m.b) / 2


def traverse_graph(atoms, kcutoff):
    if kcutoff is None:
        kcutoff = select_cutoff(atoms)

    rda = rank_determination.RDA(len(atoms))
    for (k, i, j, offset) in next_bond(atoms):
        if k > kcutoff:
            break
        rda.insert_bond(i, j, offset)

    rda.check()
    return rda.graph.find_all(), rda.all_visited, rda.ranks


def build_supercomponent(atoms, components, k, v, anchor=True):
    # build supercomponent by mapping components into visited cells
    positions = []
    numbers = []

    for c, offset in dict(v[::-1]).items():
        indices = np.where(components == c)[0]
        ps = atoms.positions + np.dot(offset, atoms.get_cell())
        positions.extend(ps[indices])
        numbers.extend(atoms.numbers[indices])
    positions = np.array(positions)
    numbers = np.array(numbers)

    # select an 'anchor' atom, which will lie at the origin
    anchor_index = next(i for i in range(len(atoms)) if components[i] == k)
    if anchor:
        positions -= atoms.positions[anchor_index]
    return positions, numbers


def select_chain_rotation(scaled):

    best = (-1, [1, 0, 0])
    for s in scaled:
        vhat = np.array([s[0], s[1], 0])
        norm = np.linalg.norm(vhat)
        if norm < 1E-6:
            continue
        vhat /= norm
        obj = np.sum(np.dot(scaled, vhat)**2)
        best = max(best, (obj, vhat), key=lambda x: x[0])
    _, vhat = best
    cost, sint, _ = vhat
    rot = np.array([[cost, -sint, 0], [sint, cost, 0], [0, 0, 1]])
    return np.dot(scaled, rot)


def isolate_chain(atoms, components, k, v):

    # identify the vector along the chain; this is the new cell vector
    basis_points = np.array([offset for c, offset in v if c == k])
    assert len(basis_points) >= 2
    assert (0, 0, 0) in [tuple(e) for e in basis_points]

    sizes = np.linalg.norm(basis_points, axis=1)
    index = np.argsort(sizes)[1]
    basis = basis_points[index]
    vector = np.dot(basis, atoms.get_cell())
    norm = np.linalg.norm(vector)
    vhat = vector / norm

    # project atoms into new basis
    positions, numbers = build_supercomponent(atoms, components, k, v)
    scaled = np.dot(positions, orthogonal_basis(vhat).T / norm)

    # move atoms into new cell
    scaled[:, 2] %= 1.0

    # subtract barycentre in x and y directions
    scaled[:, :2] -= np.mean(scaled, axis=0)[:2]

    # pick a good chain rotation (i.e. non-random)
    scaled = select_chain_rotation(scaled)

    # make cell large enough in x and y directions
    init_cell = norm * np.eye(3)
    pos = np.dot(scaled, init_cell)
    rmax = np.max(np.linalg.norm(pos[:, :2], axis=1))
    rmax = max(1, rmax)
    cell = np.diag([4 * rmax, 4 * rmax, norm])

    # construct a new atoms object containing the isolated chain
    return Atoms(numbers=numbers, positions=pos, cell=cell, pbc=[0, 0, 1])


def construct_inplane_basis(atoms, k, v):

    basis_points = np.array([offset for c, offset in v if c == k])
    assert len(basis_points) >= 3
    assert (0, 0, 0) in [tuple(e) for e in basis_points]

    sizes = np.linalg.norm(basis_points, axis=1)
    indices = np.argsort(sizes)
    basis_points = basis_points[indices]

    # identify primitive basis
    best = (float("inf"), None)
    for u, v in itertools.combinations(basis_points, 2):

        basis = np.array([[0, 0, 0], u, v])
        if np.linalg.matrix_rank(basis) < 2:
            continue

        a = np.dot(u, atoms.get_cell())
        b = np.dot(v, atoms.get_cell())
        norm = np.linalg.norm(np.cross(a, b))
        best = min(best, (norm, a, b), key=lambda x: x[0])
    _, a, b = best
    return a, b, orthogonal_basis(a, b)


def isolate_monolayer(atoms, components, k, v):

    a, b, basis = construct_inplane_basis(atoms, k, v)

    # project atoms into new basis
    c = np.cross(a, b)
    c /= np.linalg.norm(c)
    init_cell = np.dot(np.array([a, b, c]), basis.T)

    positions, numbers = build_supercomponent(atoms, components, k, v)
    scaled = np.linalg.solve(init_cell.T, np.dot(positions, basis.T).T).T

    # move atoms into new cell
    scaled[:, :2] %= 1.0

    # subtract barycentre in z-direction
    scaled[:, 2] -= np.mean(scaled, axis=0)[2]

    # make cell large enough in z-direction
    pos = np.dot(scaled, init_cell)
    zmax = np.max(np.abs(pos[:, 2]))
    cell = np.copy(init_cell)
    cell[2] *= 4 * zmax

    # construct a new atoms object containing the isolated chain
    return Atoms(numbers=numbers, positions=pos, cell=cell, pbc=[1, 1, 0])


def isolate_bulk(atoms, components, k, v):
    positions, numbers = build_supercomponent(atoms, components, k, v,
                                              anchor=False)
    atoms = Atoms(numbers=numbers, positions=positions, cell=atoms.cell,
                  pbc=[1, 1, 1])
    atoms.wrap(eps=0)
    return atoms


def isolate_cluster(atoms, components, k, v):
    positions, numbers = build_supercomponent(atoms, components, k, v)
    positions -= np.min(positions, axis=0)
    cell = np.diag(np.max(positions, axis=0))
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=False)


def isolate_components(atoms, kcutoff=None):
    """Isolates components by dimensionality type.

    Given a k-value cutoff the components (connected clusters) are
    identified.  For each component an Atoms object is created, which contains
    that component only.  The geometry of the resulting Atoms object depends
    on the component dimensionality type:

        0D: The cell is a tight box around the atoms.  pbc=[0, 0, 0].
            The cell has no physical meaning.

        1D: The chain is aligned along the z-axis.  pbc=[0, 0, 1].
            The x and y cell directions have no physical meaning.

        2D: The layer is aligned in the x-y plane.  pbc=[1, 1, 0].
            The z cell direction has no physical meaning.

        3D: The original cell is used. pbc=[1, 1, 1].

    Parameters:

    atoms: ASE atoms object
        The system to analyze.
    kcutoff: float
        The k-value cutoff to use.  Default=None, in which case the
        dimensionality scoring parameter is used to select the cutoff.

    Returns:

    components: dict
        key: the component dimenionalities.
        values: a list of Atoms objects for each dimensionality type.
    """
    data = {}
    components, all_visited, ranks = traverse_graph(atoms, kcutoff)

    for k, v in all_visited.items():
        v = sorted(list(v))

        # identify the components which constitute the component
        key = tuple(np.unique([c for c, offset in v]))
        dim = ranks[k]

        if dim == 0:
            data[('0D', key)] = isolate_cluster(atoms, components, k, v)
        elif dim == 1:
            data[('1D', key)] = isolate_chain(atoms, components, k, v)
        elif dim == 2:
            data[('2D', key)] = isolate_monolayer(atoms, components, k, v)
        elif dim == 3:
            data[('3D', key)] = isolate_bulk(atoms, components, k, v)

    result = collections.defaultdict(list)
    for (dim, _), atoms in data.items():
        result[dim].append(atoms)
    return result
