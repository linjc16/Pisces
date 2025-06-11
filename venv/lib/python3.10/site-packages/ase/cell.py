# fmt: off

from typing import Mapping, Sequence, Union

import numpy as np

import ase
from ase.utils import pbc2pbc
from ase.utils.arraywrapper import arraylike

__all__ = ['Cell']


@arraylike
class Cell:
    """Parallel epipedal unit cell of up to three dimensions.

    This object resembles a 3x3 array whose [i, j]-th element is the jth
    Cartesian coordinate of the ith unit vector.

    Cells of less than three dimensions are represented by placeholder
    unit vectors that are zero."""

    ase_objtype = 'cell'  # For JSON'ing

    def __init__(self, array):
        """Create cell.

        Parameters:

        array: 3x3 arraylike object
          The three cell vectors: cell[0], cell[1], and cell[2].
        """
        array = np.asarray(array, dtype=float)
        assert array.shape == (3, 3)
        self.array = array

    def cellpar(self, radians=False):
        """Get unit cell parameters. Sequence of 6 numbers.

        First three are unit cell vector lengths and second three
        are angles between them::

            [len(a), len(b), len(c), angle(b,c), angle(a,c), angle(a,b)]

        in degrees.

        See also :func:`ase.geometry.cell.cell_to_cellpar`."""
        from ase.geometry.cell import cell_to_cellpar
        return cell_to_cellpar(self.array, radians)

    def todict(self):
        return dict(array=self.array)

    @classmethod
    def ascell(cls, cell):
        """Return argument as a Cell object.  See :meth:`ase.cell.Cell.new`.

        A new Cell object is created if necessary."""
        if isinstance(cell, cls):
            return cell
        return cls.new(cell)

    @classmethod
    def new(cls, cell=None):
        """Create new cell from any parameters.

        If cell is three numbers, assume three lengths with right angles.

        If cell is six numbers, assume three lengths, then three angles.

        If cell is 3x3, assume three cell vectors."""

        if cell is None:
            cell = np.zeros((3, 3))

        cell = np.array(cell, float)

        if cell.shape == (3,):
            cell = np.diag(cell)
        elif cell.shape == (6,):
            from ase.geometry.cell import cellpar_to_cell
            cell = cellpar_to_cell(cell)
        elif cell.shape != (3, 3):
            raise ValueError('Cell must be length 3 sequence, length 6 '
                             'sequence or 3x3 matrix!')

        cellobj = cls(cell)
        return cellobj

    @classmethod
    def fromcellpar(cls, cellpar, ab_normal=(0, 0, 1), a_direction=None):
        """Return new Cell from cell lengths and angles.

        See also :func:`~ase.geometry.cell.cellpar_to_cell()`."""
        from ase.geometry.cell import cellpar_to_cell
        cell = cellpar_to_cell(cellpar, ab_normal, a_direction)
        return cls(cell)

    def get_bravais_lattice(self, eps=2e-4, *, pbc=True):
        """Return :class:`~ase.lattice.BravaisLattice` for this cell:

        >>> from ase.cell import Cell

        >>> cell = Cell.fromcellpar([4, 4, 4, 60, 60, 60])
        >>> print(cell.get_bravais_lattice())
        FCC(a=5.65685)

        .. note:: The Bravais lattice object follows the AFlow
           conventions.  ``cell.get_bravais_lattice().tocell()`` may
           differ from the original cell by a permutation or other
           operation which maps it to the AFlow convention.  For
           example, the orthorhombic lattice enforces a < b < c.

           To build a bandpath for a particular cell, use
           :meth:`ase.cell.Cell.bandpath` instead of this method.
           This maps the kpoints back to the original input cell.

        """
        from ase.lattice import identify_lattice
        pbc = self.mask() & pbc2pbc(pbc)
        lat, _op = identify_lattice(self, eps=eps, pbc=pbc)
        return lat

    def bandpath(
            self,
            path: str = None,
            npoints: int = None,
            *,
            density: float = None,
            special_points: Mapping[str, Sequence[float]] = None,
            eps: float = 2e-4,
            pbc: Union[bool, Sequence[bool]] = True
    ) -> "ase.dft.kpoints.BandPath":
        """Build a :class:`~ase.dft.kpoints.BandPath` for this cell.

        If special points are None, determine the Bravais lattice of
        this cell and return a suitable Brillouin zone path with
        standard special points.

        If special special points are given, interpolate the path
        directly from the available data.

        Parameters:

        path: string
            String of special point names defining the path, e.g. 'GXL'.
        npoints: int
            Number of points in total.  Note that at least one point
            is added for each special point in the path.
        density: float
            density of kpoints along the path in Å⁻¹.
        special_points: dict
            Dictionary mapping special points to scaled kpoint coordinates.
            For example ``{'G': [0, 0, 0], 'X': [1, 0, 0]}``.
        eps: float
            Tolerance for determining Bravais lattice.
        pbc: three bools
            Whether cell is periodic in each direction.  Normally not
            necessary.  If cell has three nonzero cell vectors, use
            e.g. pbc=[1, 1, 0] to request a 2D bandpath nevertheless.

        Example
        -------

        >>> from ase.cell import Cell

        >>> cell = Cell.fromcellpar([4, 4, 4, 60, 60, 60])
        >>> cell.bandpath('GXW', npoints=20)
        BandPath(path='GXW', cell=[3x3], special_points={GKLUWX}, kpts=[20x3])

        """
        # TODO: Combine with the rotation transformation from bandpath()

        cell = self.uncomplete(pbc)

        if special_points is None:
            from ase.lattice import identify_lattice
            lat, op = identify_lattice(cell, eps=eps)
            bandpath = lat.bandpath(path, npoints=npoints, density=density)
            return bandpath.transform(op)
        else:
            from ase.dft.kpoints import BandPath, resolve_custom_points
            path, special_points = resolve_custom_points(
                path, special_points, eps=eps)
            bandpath = BandPath(cell, path=path, special_points=special_points)
            return bandpath.interpolate(npoints=npoints, density=density)

    def uncomplete(self, pbc):
        """Return new cell, zeroing cell vectors where not periodic."""
        _pbc = np.empty(3, bool)
        _pbc[:] = pbc
        cell = self.copy()
        cell[~_pbc] = 0
        return cell

    def complete(self):
        """Convert missing cell vectors into orthogonal unit vectors."""
        from ase.geometry.cell import complete_cell
        return Cell(complete_cell(self.array))

    def copy(self):
        """Return a copy of this cell."""
        return Cell(self.array.copy())

    def mask(self):
        """Boolean mask of which cell vectors are nonzero."""
        return self.any(1)

    @property
    def rank(self) -> int:
        """"Return the dimension of the cell.

        Equal to the number of nonzero lattice vectors."""
        # The name ndim clashes with ndarray.ndim
        return sum(self.mask())

    @property
    def orthorhombic(self) -> bool:
        """Return whether this cell is represented by a diagonal matrix."""
        from ase.geometry.cell import is_orthorhombic
        return is_orthorhombic(self)

    def lengths(self):
        """Return the length of each lattice vector as an array."""
        return np.linalg.norm(self, axis=1)

    def angles(self):
        """Return an array with the three angles alpha, beta, and gamma."""
        return self.cellpar()[3:].copy()

    def __array__(self, dtype=None, copy=False):
        return np.array(self.array, dtype=dtype, copy=copy)

    def __bool__(self):
        return bool(self.any())  # need to convert from np.bool_

    @property
    def volume(self) -> float:
        """Get the volume of this cell.

        If there are less than 3 lattice vectors, return 0."""
        # Fail or 0 for <3D cells?
        # Definitely 0 since this is currently a property.
        # I think normally it is more convenient just to get zero
        return np.abs(np.linalg.det(self))

    @property
    def handedness(self) -> int:
        """Sign of the determinant of the matrix of cell vectors.

        1 for right-handed cells, -1 for left, and 0 for cells that
        do not span three dimensions."""
        return int(np.sign(np.linalg.det(self)))

    def scaled_positions(self, positions) -> np.ndarray:
        """Calculate scaled positions from Cartesian positions.

        The scaled positions are the positions given in the basis
        of the cell vectors.  For the purpose of defining the basis, cell
        vectors that are zero will be replaced by unit vectors as per
        :meth:`~ase.cell.Cell.complete`."""
        return np.linalg.solve(self.complete().T, np.transpose(positions)).T

    def cartesian_positions(self, scaled_positions) -> np.ndarray:
        """Calculate Cartesian positions from scaled positions."""
        return scaled_positions @ self.complete()

    def reciprocal(self) -> 'Cell':
        """Get reciprocal lattice as a Cell object.

        The reciprocal cell is defined such that

            cell.reciprocal() @ cell.T == np.diag(cell.mask())

        within machine precision.

        Does not include factor of 2 pi."""
        icell = Cell(np.linalg.pinv(self).transpose())
        icell[~self.mask()] = 0.0  # type: ignore[index]
        return icell

    def normal(self, i):
        """Normal vector of the two vectors with index different from i.

        This is the cross product of those vectors in cyclic order from i."""
        return np.cross(self[i - 2], self[i - 1])

    def normals(self):
        """Normal vectors of each axis as a 3x3 matrix."""
        return np.array([self.normal(i) for i in range(3)])

    def area(self, i):
        """Area spanned by the two vectors with index different from i."""
        return np.linalg.norm(self.normal(i))

    def areas(self):
        """Areas spanned by cell vector pairs (1, 2), (2, 0), and (0, 2)."""
        return np.linalg.norm(self.normals(), axis=1)

    def __repr__(self):
        if self.orthorhombic:
            numbers = self.lengths().tolist()
        else:
            numbers = self.tolist()

        return f'Cell({numbers})'

    def niggli_reduce(self, eps=1e-5):
        """Niggli reduce this cell, returning a new cell and mapping.

        See also :func:`ase.build.tools.niggli_reduce_cell`."""
        from ase.build.tools import niggli_reduce_cell
        cell, op = niggli_reduce_cell(self, epsfactor=eps)
        result = Cell(cell)
        return result, op

    def minkowski_reduce(self):
        """Minkowski-reduce this cell, returning new cell and mapping.

        See also :func:`ase.geometry.minkowski_reduction.minkowski_reduce`."""
        from ase.geometry.minkowski_reduction import minkowski_reduce
        cell, op = minkowski_reduce(self, self.mask())
        result = Cell(cell)
        return result, op

    def permute_axes(self, permutation):
        """Permute axes of cell."""
        assert (np.sort(permutation) == np.arange(3)).all()
        permuted = Cell(self[permutation][:, permutation])
        return permuted

    def standard_form(self, form='lower'):
        """Rotate axes such that unit cell is lower/upper triangular. The cell
        handedness is preserved.

        A lower-triangular cell with positive diagonal entries is a canonical
        (i.e. unique) description. For a left-handed cell the diagonal entries
        are negative.

        Parameters:

        form: str
            'lower' or 'upper' triangular form. The default is 'lower'.

        Returns:

        rcell: the standardized cell object

        Q: ndarray
            The orthogonal transformation.  Here, rcell @ Q = cell, where cell
            is the input cell and rcell is the lower triangular (output) cell.
        """

        sign = self.handedness
        if sign == 0:
            sign = 1

        # LQ decomposition provides an axis-aligned description of the cell.
        # Q is an orthogonal matrix and L is a lower triangular matrix. The
        # decomposition is a unique description if the diagonal elements are
        # all positive (negative for a left-handed cell).
        if form == 'lower':
            Q, L = np.linalg.qr(self.T)
            Q = Q.T
            L = L.T
        elif form == 'upper':
            Q, L = np.linalg.qr(self.T[::-1, ::-1])
            Q = Q.T[::-1, ::-1]
            L = L.T[::-1, ::-1]
        else:
            raise ValueError('form must be either "lower" or "upper"')

        # correct the signs of the diagonal elements
        signs = np.sign(np.diag(L))
        indices = np.where(signs == 0)[0]
        signs[indices] = 1
        indices = np.where(signs != sign)[0]
        L[:, indices] *= -1
        Q[indices] *= -1
        return Cell(L), Q

    # XXX We want a reduction function that brings the cell into
    # standard form as defined by Setyawan and Curtarolo.
