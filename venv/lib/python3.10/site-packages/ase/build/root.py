# fmt: off

from math import atan2, cos, log10, sin

import numpy as np


def hcp0001_root(symbol, root, size, a=None, c=None,
                 vacuum=None, orthogonal=False):
    """HCP(0001) surface maniupulated to have a x unit side length
    of *root* before repeating.  This also results in *root* number
    of repetitions of the cell.


    The first 20 valid roots for nonorthogonal are...
    1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25,
    27, 28, 31, 36, 37, 39, 43, 48, 49"""
    from ase.build import hcp0001
    atoms = hcp0001(symbol=symbol, size=(1, 1, size[2]),
                    a=a, c=c, vacuum=vacuum, orthogonal=orthogonal)
    atoms = root_surface(atoms, root)
    atoms *= (size[0], size[1], 1)
    return atoms


def fcc111_root(symbol, root, size, a=None,
                vacuum=None, orthogonal=False):
    """FCC(111) surface maniupulated to have a x unit side length
    of *root* before repeating. This also results in *root* number
    of repetitions of the cell.

    The first 20 valid roots for nonorthogonal are...
    1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25, 27,
    28, 31, 36, 37, 39, 43, 48, 49"""
    from ase.build import fcc111
    atoms = fcc111(symbol=symbol, size=(1, 1, size[2]),
                   a=a, vacuum=vacuum, orthogonal=orthogonal)
    atoms = root_surface(atoms, root)
    atoms *= (size[0], size[1], 1)
    return atoms


def bcc111_root(symbol, root, size, a=None,
                vacuum=None, orthogonal=False):
    """BCC(111) surface maniupulated to have a x unit side length
    of *root* before repeating. This also results in *root* number
    of repetitions of the cell.


    The first 20 valid roots for nonorthogonal are...
    1, 3, 4, 7, 9, 12, 13, 16, 19, 21, 25,
    27, 28, 31, 36, 37, 39, 43, 48, 49"""
    from ase.build import bcc111
    atoms = bcc111(symbol=symbol, size=(1, 1, size[2]),
                   a=a, vacuum=vacuum, orthogonal=orthogonal)
    atoms = root_surface(atoms, root)
    atoms *= (size[0], size[1], 1)
    return atoms


def point_in_cell_2d(point, cell, eps=1e-8):
    """This function takes a 2D slice of the cell in the XY plane and calculates
    if a point should lie in it.  This is used as a more accurate method of
    ensuring we find all of the correct cell repetitions in the root surface
    code.  The Z axis is totally ignored but for most uses this should be fine.
    """
    # Define area of a triangle
    def tri_area(t1, t2, t3):
        t1x, t1y = t1[0:2]
        t2x, t2y = t2[0:2]
        t3x, t3y = t3[0:2]
        return abs(t1x * (t2y - t3y) + t2x *
                   (t3y - t1y) + t3x * (t1y - t2y)) / 2

    # c0, c1, c2, c3 define a parallelogram
    c0 = (0, 0)
    c1 = cell[0, 0:2]
    c2 = cell[1, 0:2]
    c3 = c1 + c2

    # Get area of parallelogram
    cA = tri_area(c0, c1, c2) + tri_area(c1, c2, c3)

    # Get area of triangles formed from adjacent vertices of parallelogram and
    # point in question.
    pA = tri_area(point, c0, c1) + tri_area(point, c1, c2) + \
        tri_area(point, c2, c3) + tri_area(point, c3, c0)

    # If combined area of triangles from point is larger than area of
    # parallelogram, point is not inside parallelogram.
    return pA <= cA + eps


def _root_cell_normalization(primitive_slab):
    """Returns the scaling factor for x axis and cell normalized by that
    factor"""

    xscale = np.linalg.norm(primitive_slab.cell[0, 0:2])
    cell_vectors = primitive_slab.cell[0:2, 0:2] / xscale
    return xscale, cell_vectors


def _root_surface_analysis(primitive_slab, root, eps=1e-8):
    """A tool to analyze a slab and look for valid roots that exist, up to
    the given root. This is useful for generating all possible cells
    without prior knowledge.

    *primitive slab* is the primitive cell to analyze.

    *root* is the desired root to find, and all below.

    This is the internal function which gives extra data to root_surface.
    """

    # Setup parameters for cell searching
    logeps = int(-log10(eps))
    _xscale, cell_vectors = _root_cell_normalization(primitive_slab)

    # Allocate grid for cell search search
    points = np.indices((root + 1, root + 1)).T.reshape(-1, 2)

    # Find points corresponding to full cells
    cell_points = [cell_vectors[0] * x + cell_vectors[1] * y for x, y in points]

    # Find point close to the desired cell (floating point error possible)
    roots = np.around(np.linalg.norm(cell_points, axis=1)**2, logeps)

    valid_roots = np.nonzero(roots == root)[0]
    if len(valid_roots) == 0:
        raise ValueError(
            "Invalid root {} for cell {}".format(
                root, cell_vectors))
    int_roots = np.array([int(this_root) for this_root in roots
                          if this_root.is_integer() and this_root <= root])
    return cell_points, cell_points[np.nonzero(
        roots == root)[0][0]], set(int_roots[1:])


def root_surface_analysis(primitive_slab, root, eps=1e-8):
    """A tool to analyze a slab and look for valid roots that exist, up to
    the given root. This is useful for generating all possible cells
    without prior knowledge.

    *primitive slab* is the primitive cell to analyze.

    *root* is the desired root to find, and all below."""
    return _root_surface_analysis(
        primitive_slab=primitive_slab, root=root, eps=eps)[2]


def root_surface(primitive_slab, root, eps=1e-8):
    """Creates a cell from a primitive cell that repeats along the x and y
    axis in a way consisent with the primitive cell, that has been cut
    to have a side length of *root*.

    *primitive cell* should be a primitive 2d cell of your slab, repeated
    as needed in the z direction.

    *root* should be determined using an analysis tool such as the
    root_surface_analysis function, or prior knowledge. It should always
    be a whole number as it represents the number of repetitions."""

    atoms = primitive_slab.copy()

    xscale, cell_vectors = _root_cell_normalization(primitive_slab)

    # Do root surface analysis
    cell_points, root_point, _roots = _root_surface_analysis(
        primitive_slab, root, eps=eps)

    # Find new cell
    root_angle = -atan2(root_point[1], root_point[0])
    root_rotation = [[cos(root_angle), -sin(root_angle)],
                     [sin(root_angle), cos(root_angle)]]
    root_scale = np.linalg.norm(root_point)

    cell = np.array([np.dot(x, root_rotation) *
                    root_scale for x in cell_vectors])

    # Find all cell centers within the cell
    shift = cell_vectors.sum(axis=0) / 2
    cell_points = [
        point for point in cell_points if point_in_cell_2d(
            point + shift, cell, eps=eps)]

    # Setup new cell
    atoms.rotate(root_angle, v="z")
    atoms *= (root, root, 1)
    atoms.cell[0:2, 0:2] = cell * xscale
    atoms.center()

    # Remove all extra atoms
    del atoms[[atom.index for atom in atoms if not point_in_cell_2d(
        atom.position, atoms.cell, eps=eps)]]

    # Rotate cell back to original orientation
    standard_rotation = [[cos(-root_angle), -sin(-root_angle), 0],
                         [sin(-root_angle), cos(-root_angle), 0],
                         [0, 0, 1]]

    new_cell = np.array([np.dot(x, standard_rotation) for x in atoms.cell])
    new_positions = np.array([np.dot(x, standard_rotation)
                              for x in atoms.positions])

    atoms.cell = new_cell
    atoms.positions = new_positions
    return atoms
