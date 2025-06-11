# fmt: off

"""Helper functions for creating supercells."""

import warnings

import numpy as np

from ase import Atoms


class SupercellError(Exception):
    """Use if construction of supercell fails"""


def get_deviation_from_optimal_cell_shape(cell, target_shape="sc", norm=None):
    r"""Calculate the deviation from the target cell shape.

    Calculates the deviation of the given cell metric from the ideal
    cell metric defining a certain shape. Specifically, the function
    evaluates the expression `\Delta = || Q \mathbf{h} -
    \mathbf{h}_{target}||_2`, where `\mathbf{h}` is the input
    metric (*cell*) and `Q` is a normalization factor (*norm*)
    while the target metric `\mathbf{h}_{target}` (via
    *target_shape*) represent simple cubic ('sc') or face-centered
    cubic ('fcc') cell shapes.

    Replaced with code from the `doped` defect simulation package
    (https://doped.readthedocs.io) to be rotationally invariant,
    boosting performance.

    Parameters
    ----------
    cell : (..., 3, 3) array_like
        Metric given as a 3x3 matrix of the input structure.
        Multiple cells can also be given as a higher-dimensional array.
    target_shape : {'sc', 'fcc'}
        Desired supercell shape. Can be 'sc' for simple cubic or
        'fcc' for face-centered cubic.
    norm : float
        Specify the normalization factor. This is useful to avoid
        recomputing the normalization factor when computing the
        deviation for a series of P matrices.

    Returns
    -------
    float or ndarray
        Cell metric(s) (0 is perfect score)

    .. deprecated:: 3.24.0
        `norm` is unused in ASE 3.24.0 and removed in ASE 3.25.0.

    """
    if norm is not None:
        warnings.warn(
            '`norm` is unused in ASE 3.24.0 and removed in ASE 3.25.0',
            FutureWarning,
        )

    cell = np.asarray(cell)
    cell_lengths = np.sqrt(np.add.reduce(cell**2, axis=-1))
    eff_cubic_length = np.cbrt(np.abs(np.linalg.det(cell)))  # 'a_0'

    if target_shape == 'sc':
        target_length = eff_cubic_length

    elif target_shape == 'fcc':
        # FCC is characterised by 60 degree angles & lattice vectors = 2**(1/6)
        # times the eff cubic length:
        target_length = eff_cubic_length * 2 ** (1 / 6)

    else:
        raise ValueError(target_shape)

    inv_target_length = 1.0 / target_length

    # rms difference to eff cubic/FCC length:
    diffs = cell_lengths * inv_target_length[..., None] - 1.0
    return np.sqrt(np.add.reduce(diffs**2, axis=-1))


def find_optimal_cell_shape(
    cell,
    target_size,
    target_shape,
    lower_limit=-2,
    upper_limit=2,
    verbose=False,
):
    """Obtain the optimal transformation matrix for a supercell of target size
    and shape.

    Returns the transformation matrix that produces a supercell
    corresponding to *target_size* unit cells with metric *cell* that
    most closely approximates the shape defined by *target_shape*.

    Updated with code from the `doped` defect simulation package
    (https://doped.readthedocs.io) to be rotationally invariant and
    allow transformation matrices with negative determinants, boosting
    performance.

    Parameters:

    cell: 2D array of floats
        Metric given as a (3x3 matrix) of the input structure.
    target_size: integer
        Size of desired supercell in number of unit cells.
    target_shape: str
        Desired supercell shape. Can be 'sc' for simple cubic or
        'fcc' for face-centered cubic.
    lower_limit: int
        Lower limit of search range.
    upper_limit: int
        Upper limit of search range.
    verbose: bool
        Set to True to obtain additional information regarding
        construction of transformation matrix.

    Returns:
        2D array of integers: Transformation matrix that produces the
        optimal supercell.
    """
    cell = np.asarray(cell)

    # Set up target metric
    if target_shape == 'sc':
        target_metric = np.eye(3)
    elif target_shape == 'fcc':
        target_metric = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]],
                                       dtype=float)
    else:
        raise ValueError(target_shape)

    if verbose:
        print("target metric (h_target):")
        print(target_metric)

    # Normalize cell metric to reduce computation time during looping
    norm = (target_size * abs(np.linalg.det(cell)) /
            np.linalg.det(target_metric)) ** (-1.0 / 3)
    norm_cell = norm * cell
    if verbose:
        print(f"normalization factor (Q): {norm:g}")

    # Approximate initial P matrix
    ideal_P = np.dot(target_metric, np.linalg.inv(norm_cell))
    if verbose:
        print("idealized transformation matrix:")
        print(ideal_P)
    starting_P = np.array(np.around(ideal_P, 0), dtype=int)
    if verbose:
        print("closest integer transformation matrix (P_0):")
        print(starting_P)

    # Build a big matrix of all admissible integer matrix operations.
    # (If this takes too much memory we could do blocking but there are
    # too many for looping one by one.)
    dimensions = [(upper_limit + 1) - lower_limit] * 9
    operations = np.moveaxis(np.indices(dimensions), 0, -1).reshape(-1, 3, 3)
    operations += lower_limit  # Each element runs from lower to upper limits.
    operations += starting_P
    determinants = np.linalg.det(operations)

    # screen supercells with the target size
    good_indices = np.where(abs(determinants - target_size) < 1e-12)[0]
    if not good_indices.size:
        print("Failed to find a transformation matrix.")
        return None
    operations = operations[good_indices]

    # evaluate derivations of the screened supercells
    scores = get_deviation_from_optimal_cell_shape(
        operations @ cell,
        target_shape,
    )
    imin = np.argmin(scores)
    best_score = scores[imin]

    # screen candidates with the same best score
    operations = operations[np.abs(scores - best_score) < 1e-6]

    # select the one whose cell orientation is the closest to the target
    # https://gitlab.com/ase/ase/-/merge_requests/3522
    imin = np.argmin(np.add.reduce((operations - ideal_P)**2, axis=(-2, -1)))
    optimal_P = operations[imin]

    if np.linalg.det(optimal_P) <= 0:
        optimal_P *= -1  # flip signs if negative determinant

    # Finalize.
    if verbose:
        print(f"smallest score (|Q P h_p - h_target|_2): {best_score:f}")
        print("optimal transformation matrix (P_opt):")
        print(optimal_P)
        print("supercell metric:")
        print(np.round(np.dot(optimal_P, cell), 4))
        det = np.linalg.det(optimal_P)
        print(f"determinant of optimal transformation matrix: {det:g}")
    return optimal_P


def make_supercell(prim, P, *, wrap=True, order="cell-major", tol=1e-5):
    r"""Generate a supercell by applying a general transformation (*P*) to
    the input configuration (*prim*).

    The transformation is described by a 3x3 integer matrix
    `\mathbf{P}`. Specifically, the new cell metric
    `\mathbf{h}` is given in terms of the metric of the input
    configuration `\mathbf{h}_p` by `\mathbf{P h}_p =
    \mathbf{h}`.

    Parameters:

    prim: ASE Atoms object
        Input configuration.
    P: 3x3 integer matrix
        Transformation matrix `\mathbf{P}`.
    wrap: bool
        wrap in the end
    order: str (default: "cell-major")
        how to order the atoms in the supercell

        "cell-major":
        [atom1_shift1, atom2_shift1, ..., atom1_shift2, atom2_shift2, ...]
        i.e. run first over all the atoms in cell1 and then move to cell2.

        "atom-major":
        [atom1_shift1, atom1_shift2, ..., atom2_shift1, atom2_shift2, ...]
        i.e. run first over atom1 in all the cells and then move to atom2.
        This may be the order preferred by most VASP users.

    tol: float
        tolerance for wrapping
    """

    supercell_matrix = P
    supercell = clean_matrix(supercell_matrix @ prim.cell)

    # cartesian lattice points
    lattice_points_frac = lattice_points_in_supercell(supercell_matrix)
    lattice_points = np.dot(lattice_points_frac, supercell)
    N = len(lattice_points)

    if order == "cell-major":
        shifted = prim.positions[None, :, :] + lattice_points[:, None, :]
    elif order == "atom-major":
        shifted = prim.positions[:, None, :] + lattice_points[None, :, :]
    else:
        raise ValueError(f"invalid order: {order}")
    shifted_reshaped = shifted.reshape(-1, 3)

    superatoms = Atoms(positions=shifted_reshaped,
                       cell=supercell,
                       pbc=prim.pbc)

    # Copy over any other possible arrays, inspired by atoms.__imul__
    for name, arr in prim.arrays.items():
        if name == "positions":
            # This was added during construction of the super cell
            continue
        shape = (N * arr.shape[0], *arr.shape[1:])
        if order == "cell-major":
            new_arr = np.repeat(arr[None, :], N, axis=0).reshape(shape)
        elif order == "atom-major":
            new_arr = np.repeat(arr[:, None], N, axis=1).reshape(shape)
        superatoms.set_array(name, new_arr)

    # check number of atoms is correct
    n_target = abs(int(np.round(np.linalg.det(supercell_matrix) * len(prim))))
    if n_target != len(superatoms):
        msg = "Number of atoms in supercell: {}, expected: {}".format(
            n_target, len(superatoms))
        raise SupercellError(msg)

    if wrap:
        superatoms.wrap(eps=tol)

    return superatoms


def lattice_points_in_supercell(supercell_matrix):
    """Find all lattice points contained in a supercell.

    Adapted from pymatgen, which is available under MIT license:
    The MIT License (MIT) Copyright (c) 2011-2012 MIT & The Regents of the
    University of California, through Lawrence Berkeley National Laboratory
    """

    diagonals = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])
    d_points = np.dot(diagonals, supercell_matrix)

    mins = np.min(d_points, axis=0)
    maxes = np.max(d_points, axis=0) + 1

    ar = np.arange(mins[0], maxes[0])[:, None] * np.array([1, 0, 0])[None, :]
    br = np.arange(mins[1], maxes[1])[:, None] * np.array([0, 1, 0])[None, :]
    cr = np.arange(mins[2], maxes[2])[:, None] * np.array([0, 0, 1])[None, :]

    all_points = ar[:, None, None] + br[None, :, None] + cr[None, None, :]
    all_points = all_points.reshape((-1, 3))

    frac_points = np.dot(all_points, np.linalg.inv(supercell_matrix))

    tvects = frac_points[np.all(frac_points < 1 - 1e-10, axis=1)
                         & np.all(frac_points >= -1e-10, axis=1)]
    assert len(tvects) == round(abs(np.linalg.det(supercell_matrix)))
    return tvects


def clean_matrix(matrix, eps=1e-12):
    """ clean from small values"""
    matrix = np.array(matrix)
    for ij in np.ndindex(matrix.shape):
        if abs(matrix[ij]) < eps:
            matrix[ij] = 0
    return matrix
