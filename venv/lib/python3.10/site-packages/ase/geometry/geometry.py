# fmt: off

# Copyright (C) 2010, Jesper Friis
# (see accompanying license files for details).

"""Utility tools for atoms/geometry manipulations.
   - convenient creation of slabs and interfaces of
different orientations.
   - detection of duplicate atoms / atoms within cutoff radius
"""

import itertools

import numpy as np

from ase.cell import Cell
from ase.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.utils import pbc2pbc


def translate_pretty(fractional, pbc):
    """Translates atoms such that fractional positions are minimized."""

    for i in range(3):
        if not pbc[i]:
            continue

        indices = np.argsort(fractional[:, i])
        sp = fractional[indices, i]

        widths = (np.roll(sp, 1) - sp) % 1.0
        fractional[:, i] -= sp[np.argmin(widths)]
        fractional[:, i] %= 1.0
    return fractional


def wrap_positions(positions, cell, pbc=True, center=(0.5, 0.5, 0.5),
                   pretty_translation=False, eps=1e-7):
    """Wrap positions to unit cell.

    Returns positions changed by a multiple of the unit cell vectors to
    fit inside the space spanned by these vectors.  See also the
    :meth:`ase.Atoms.wrap` method.

    Parameters:

    positions: float ndarray of shape (n, 3)
        Positions of the atoms
    cell: float ndarray of shape (3, 3)
        Unit cell vectors.
    pbc: one or 3 bool
        For each axis in the unit cell decides whether the positions
        will be moved along this axis.
    center: three float
        The positons in fractional coordinates that the new positions
        will be nearest possible to.
    pretty_translation: bool
        Translates atoms such that fractional coordinates are minimized.
    eps: float
        Small number to prevent slightly negative coordinates from being
        wrapped.

    Example:

    >>> from ase.geometry import wrap_positions
    >>> wrap_positions([[-0.1, 1.01, -0.5]],
    ...                [[1, 0, 0], [0, 1, 0], [0, 0, 4]],
    ...                pbc=[1, 1, 0])
    array([[ 0.9 ,  0.01, -0.5 ]])
    """

    if not hasattr(center, '__len__'):
        center = (center,) * 3

    pbc = pbc2pbc(pbc)
    shift = np.asarray(center) - 0.5 - eps

    # Don't change coordinates when pbc is False
    shift[np.logical_not(pbc)] = 0.0

    assert np.asarray(cell)[np.asarray(pbc)].any(axis=1).all(), (cell, pbc)

    cell = complete_cell(cell)
    fractional = np.linalg.solve(cell.T,
                                 np.asarray(positions).T).T - shift

    if pretty_translation:
        fractional = translate_pretty(fractional, pbc)
        shift = np.asarray(center) - 0.5
        shift[np.logical_not(pbc)] = 0.0
        fractional += shift
    else:
        for i, periodic in enumerate(pbc):
            if periodic:
                fractional[:, i] %= 1.0
                fractional[:, i] += shift[i]

    return np.dot(fractional, cell)


def get_layers(atoms, miller, tolerance=0.001):
    """Returns two arrays describing which layer each atom belongs
    to and the distance between the layers and origo.

    Parameters:

    miller: 3 integers
        The Miller indices of the planes. Actually, any direction
        in reciprocal space works, so if a and b are two float
        vectors spanning an atomic plane, you can get all layers
        parallel to this with miller=np.cross(a,b).
    tolerance: float
        The maximum distance in Angstrom along the plane normal for
        counting two atoms as belonging to the same plane.

    Returns:

    tags: array of integres
        Array of layer indices for each atom.
    levels: array of floats
        Array of distances in Angstrom from each layer to origo.

    Example:

    >>> import numpy as np

    >>> from ase.spacegroup import crystal
    >>> from ase.geometry.geometry import get_layers

    >>> atoms = crystal('Al', [(0,0,0)], spacegroup=225, cellpar=4.05)
    >>> np.round(atoms.positions, decimals=5)  # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.   ,  0.   ,  0.   ],
           [ 0.   ,  2.025,  2.025],
           [ 2.025,  0.   ,  2.025],
           [ 2.025,  2.025,  0.   ]])
    >>> get_layers(atoms, (0,0,1))  # doctest: +ELLIPSIS
    (array([0, 1, 1, 0]...), array([ 0.   ,  2.025]))
    """
    miller = np.asarray(miller)

    metric = np.dot(atoms.cell, atoms.cell.T)
    c = np.linalg.solve(metric.T, miller.T).T
    miller_norm = np.sqrt(np.dot(c, miller))
    d = np.dot(atoms.get_scaled_positions(), miller) / miller_norm

    keys = np.argsort(d)
    ikeys = np.argsort(keys)
    mask = np.concatenate(([True], np.diff(d[keys]) > tolerance))
    tags = np.cumsum(mask)[ikeys]
    if tags.min() == 1:
        tags -= 1

    levels = d[keys][mask]
    return tags, levels


def naive_find_mic(v, cell):
    """Finds the minimum-image representation of vector(s) v.
    Safe to use for (pbc.all() and (norm(v_mic) < 0.5 * min(cell.lengths()))).
    Can otherwise fail for non-orthorhombic cells.
    Described in:
    W. Smith, "The Minimum Image Convention in Non-Cubic MD Cells", 1989,
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.57.1696."""
    f = Cell(cell).scaled_positions(v)
    f -= np.floor(f + 0.5)
    vmin = f @ cell
    vlen = np.linalg.norm(vmin, axis=1)
    return vmin, vlen


def general_find_mic(v, cell, pbc=True):
    """Finds the minimum-image representation of vector(s) v. Using the
    Minkowski reduction the algorithm is relatively slow but safe for any cell.
    """

    cell = complete_cell(cell)
    rcell, _ = minkowski_reduce(cell, pbc=pbc)
    positions = wrap_positions(v, rcell, pbc=pbc, eps=0)

    # In a Minkowski-reduced cell we only need to test nearest neighbors,
    # or "Voronoi-relevant" vectors. These are a subset of combinations of
    # [-1, 0, 1] of the reduced cell vectors.

    # Define ranges [-1, 0, 1] for periodic directions and [0] for aperiodic
    # directions.
    ranges = [np.arange(-1 * p, p + 1) for p in pbc]

    # Get Voronoi-relevant vectors.
    # Pre-pend (0, 0, 0) to resolve issue #772
    hkls = np.array([(0, 0, 0)] + list(itertools.product(*ranges)))
    vrvecs = hkls @ rcell

    # Map positions into neighbouring cells.
    x = positions + vrvecs[:, None]

    # Find minimum images
    lengths = np.linalg.norm(x, axis=2)
    indices = np.argmin(lengths, axis=0)
    vmin = x[indices, np.arange(len(positions)), :]
    vlen = lengths[indices, np.arange(len(positions))]
    return vmin, vlen


def find_mic(v, cell, pbc=True):
    """Finds the minimum-image representation of vector(s) v using either one
    of two find mic algorithms depending on the given cell, v and pbc."""

    cell = Cell(cell)
    pbc = cell.any(1) & pbc2pbc(pbc)
    dim = np.sum(pbc)
    v = np.asarray(v)
    single = v.ndim == 1
    v = np.atleast_2d(v)

    if dim > 0:
        naive_find_mic_is_safe = False
        if dim == 3:
            vmin, vlen = naive_find_mic(v, cell)
            # naive find mic is safe only for the following condition
            if (vlen < 0.5 * min(cell.lengths())).all():
                naive_find_mic_is_safe = True  # hence skip Minkowski reduction

        if not naive_find_mic_is_safe:
            vmin, vlen = general_find_mic(v, cell, pbc=pbc)
    else:
        vmin = v.copy()
        vlen = np.linalg.norm(vmin, axis=1)

    if single:
        return vmin[0], vlen[0]
    else:
        return vmin, vlen


def conditional_find_mic(vectors, cell, pbc):
    """Return vectors and their lengths considering cell and pbc.

    The minimum image convention is applied if cell and pbc are set.
    This can be used like a simple version of get_distances.
    """
    vectors = np.array(vectors)
    if (cell is None) != (pbc is None):
        raise ValueError("cell or pbc must be both set or both be None")
    if cell is not None:
        mics = [find_mic(v, cell, pbc) for v in vectors]
        vectors, vector_lengths = zip(*mics)
    else:
        vector_lengths = np.sqrt(np.add.reduce(vectors**2, axis=-1))
    return vectors, vector_lengths


def get_angles(v0, v1, cell=None, pbc=None):
    """Get angles formed by two lists of vectors.

    Calculate angle in degrees between vectors v0 and v1

    Set a cell and pbc to enable minimum image
    convention, otherwise angles are taken as-is.
    """
    (v0, v1), (nv0, nv1) = conditional_find_mic([v0, v1], cell, pbc)

    if (nv0 <= 0).any() or (nv1 <= 0).any():
        raise ZeroDivisionError('Undefined angle')
    v0n = v0 / nv0[:, np.newaxis]
    v1n = v1 / nv1[:, np.newaxis]
    # We just normalized the vectors, but in some cases we can get
    # bad things like 1+2e-16.  These we clip away:
    angles = np.arccos(np.einsum('ij,ij->i', v0n, v1n).clip(-1.0, 1.0))
    return np.degrees(angles)


def get_angles_derivatives(v0, v1, cell=None, pbc=None):
    """Get derivatives of angles formed by two lists of vectors (v0, v1) w.r.t.
    Cartesian coordinates in degrees.

    Set a cell and pbc to enable minimum image
    convention, otherwise derivatives of angles are taken as-is.

    There is a singularity in the derivatives for sin(angle) -> 0 for which
    a ZeroDivisionError is raised.

    Derivative output format: [[dx_a0, dy_a0, dz_a0], [...], [..., dz_a2].
    """
    (v0, v1), (nv0, nv1) = conditional_find_mic([v0, v1], cell, pbc)

    angles = np.radians(get_angles(v0, v1, cell=cell, pbc=pbc))
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)
    if (sin_angles == 0.).any():  # identify singularities
        raise ZeroDivisionError('Singularity for derivative of a planar angle')

    product = nv0 * nv1
    deriv_d0 = (-(v1 / product[:, np.newaxis]  # derivatives by atom 0
                  - np.einsum('ij,i->ij', v0, cos_angles / nv0**2))
                / sin_angles[:, np.newaxis])
    deriv_d2 = (-(v0 / product[:, np.newaxis]  # derivatives by atom 2
                  - np.einsum('ij,i->ij', v1, cos_angles / nv1**2))
                / sin_angles[:, np.newaxis])
    deriv_d1 = -(deriv_d0 + deriv_d2)  # derivatives by atom 1
    derivs = np.stack((deriv_d0, deriv_d1, deriv_d2), axis=1)
    return np.degrees(derivs)


def get_dihedrals(v0, v1, v2, cell=None, pbc=None):
    """Get dihedral angles formed by three lists of vectors.

    Calculate dihedral angle (in degrees) between the vectors a0->a1,
    a1->a2 and a2->a3, written as v0, v1 and v2.

    Set a cell and pbc to enable minimum image
    convention, otherwise angles are taken as-is.
    """
    (v0, v1, v2), (_, nv1, _) = conditional_find_mic([v0, v1, v2], cell, pbc)

    v1n = v1 / nv1[:, np.newaxis]
    # v, w: projection of v0, v2 onto plane perpendicular to v1
    v = -v0 - np.einsum('ij,ij,ik->ik', -v0, v1n, v1n)
    w = v2 - np.einsum('ij,ij,ik->ik', v2, v1n, v1n)

    # formula returns 0 for undefined dihedrals; prefer ZeroDivisionError
    undefined_v = np.all(v == 0.0, axis=1)
    undefined_w = np.all(w == 0.0, axis=1)
    if np.any([undefined_v, undefined_w]):
        raise ZeroDivisionError('Undefined dihedral for planar inner angle')

    x = np.einsum('ij,ij->i', v, w)
    y = np.einsum('ij,ij->i', np.cross(v1n, v, axis=1), w)
    dihedrals = np.arctan2(y, x)            # dihedral angle in [-pi, pi]
    dihedrals[dihedrals < 0.] += 2 * np.pi  # dihedral angle in [0, 2*pi]
    return np.degrees(dihedrals)


def get_dihedrals_derivatives(v0, v1, v2, cell=None, pbc=None):
    """Get derivatives of dihedrals formed by three lists of vectors
    (v0, v1, v2) w.r.t Cartesian coordinates in degrees.

    Set a cell and pbc to enable minimum image
    convention, otherwise dihedrals are taken as-is.

    Derivative output format: [[dx_a0, dy_a0, dz_a0], ..., [..., dz_a3]].
    """
    (v0, v1, v2), (nv0, nv1, nv2) = conditional_find_mic([v0, v1, v2], cell,
                                                         pbc)

    v0n = v0 / nv0[:, np.newaxis]
    v1n = v1 / nv1[:, np.newaxis]
    v2n = v2 / nv2[:, np.newaxis]
    normal_v01 = np.cross(v0n, v1n, axis=1)
    normal_v12 = np.cross(v1n, v2n, axis=1)
    cos_psi01 = np.einsum('ij,ij->i', v0n, v1n)  # == np.sum(v0 * v1, axis=1)
    sin_psi01 = np.sin(np.arccos(cos_psi01))
    cos_psi12 = np.einsum('ij,ij->i', v1n, v2n)
    sin_psi12 = np.sin(np.arccos(cos_psi12))
    if (sin_psi01 == 0.).any() or (sin_psi12 == 0.).any():
        msg = ('Undefined derivative for undefined dihedral with planar inner '
               'angle')
        raise ZeroDivisionError(msg)

    deriv_d0 = -normal_v01 / (nv0 * sin_psi01**2)[:, np.newaxis]  # by atom 0
    deriv_d3 = normal_v12 / (nv2 * sin_psi12**2)[:, np.newaxis]  # by atom 3
    deriv_d1 = (((nv1 + nv0 * cos_psi01) / nv1)[:, np.newaxis] * -deriv_d0
                + (cos_psi12 * nv2 / nv1)[:, np.newaxis] * deriv_d3)  # by a1
    deriv_d2 = (-((nv1 + nv2 * cos_psi12) / nv1)[:, np.newaxis] * deriv_d3
                - (cos_psi01 * nv0 / nv1)[:, np.newaxis] * -deriv_d0)  # by a2
    derivs = np.stack((deriv_d0, deriv_d1, deriv_d2, deriv_d3), axis=1)
    return np.degrees(derivs)


def get_distances(p1, p2=None, cell=None, pbc=None):
    """Return distance matrix of every position in p1 with every position in p2

    If p2 is not set, it is assumed that distances between all positions in p1
    are desired. p2 will be set to p1 in this case.

    Use set cell and pbc to use the minimum image convention.
    """
    p1 = np.atleast_2d(p1)
    if p2 is None:
        np1 = len(p1)
        ind1, ind2 = np.triu_indices(np1, k=1)
        D = p1[ind2] - p1[ind1]
    else:
        p2 = np.atleast_2d(p2)
        D = (p2[np.newaxis, :, :] - p1[:, np.newaxis, :]).reshape((-1, 3))

    (D, ), (D_len, ) = conditional_find_mic([D], cell=cell, pbc=pbc)

    if p2 is None:
        Dout = np.zeros((np1, np1, 3))
        Dout[(ind1, ind2)] = D
        Dout -= np.transpose(Dout, axes=(1, 0, 2))

        Dout_len = np.zeros((np1, np1))
        Dout_len[(ind1, ind2)] = D_len
        Dout_len += Dout_len.T
        return Dout, Dout_len

    # Expand back to matrix indexing
    D.shape = (-1, len(p2), 3)
    D_len.shape = (-1, len(p2))

    return D, D_len


def get_distances_derivatives(v0, cell=None, pbc=None):
    """Get derivatives of distances for all vectors in v0 w.r.t. Cartesian
    coordinates in Angstrom.

    Set cell and pbc to use the minimum image convention.

    There is a singularity for distances -> 0 for which a ZeroDivisionError is
    raised.
    Derivative output format: [[dx_a0, dy_a0, dz_a0], [dx_a1, dy_a1, dz_a1]].
    """
    (v0, ), (dists, ) = conditional_find_mic([v0], cell, pbc)

    if (dists <= 0.).any():  # identify singularities
        raise ZeroDivisionError('Singularity for derivative of a zero distance')

    derivs_d0 = np.einsum('i,ij->ij', -1. / dists, v0)  # derivatives by atom 0
    derivs_d1 = -derivs_d0                              # derivatives by atom 1
    derivs = np.stack((derivs_d0, derivs_d1), axis=1)
    return derivs


def get_duplicate_atoms(atoms, cutoff=0.1, delete=False):
    """Get list of duplicate atoms and delete them if requested.

    Identify all atoms which lie within the cutoff radius of each other.
    Delete one set of them if delete == True.
    """
    dists = get_distances(atoms.positions, cell=atoms.cell, pbc=atoms.pbc)[1]
    dup = np.argwhere(dists < cutoff)
    dup = dup[dup[:, 0] < dup[:, 1]]  # indices at upper triangle
    if delete and dup.size != 0:
        del atoms[dup[:, 0]]
    return dup


def permute_axes(atoms, permutation):
    """Permute axes of unit cell and atom positions. Considers only cell and
    atomic positions. Other vector quantities such as momenta are not
    modified."""
    assert (np.sort(permutation) == np.arange(3)).all()

    permuted = atoms.copy()
    scaled = permuted.get_scaled_positions()
    permuted.set_cell(permuted.cell.permute_axes(permutation),
                      scale_atoms=False)
    permuted.set_scaled_positions(scaled[:, permutation])
    permuted.set_pbc(permuted.pbc[permutation])
    return permuted
