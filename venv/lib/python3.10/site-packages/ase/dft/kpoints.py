# fmt: off

import re
import warnings
from typing import Dict

import numpy as np

import ase  # Annotations
from ase.cell import Cell
from ase.utils import jsonable


def monkhorst_pack(size):
    """Construct a uniform sampling of k-space of given size."""
    if np.less_equal(size, 0).any():
        raise ValueError(f'Illegal size: {list(size)}')
    kpts = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
    return (kpts + 0.5) / size - 0.5


def get_monkhorst_pack_size_and_offset(kpts):
    """Find Monkhorst-Pack size and offset.

    Returns (size, offset), where::

        kpts = monkhorst_pack(size) + offset.

    The set of k-points must not have been symmetry reduced."""

    if len(kpts) == 1:
        return np.ones(3, int), np.array(kpts[0], dtype=float)

    size = np.zeros(3, int)
    for c in range(3):
        # Determine increment between k-points along current axis
        delta = max(np.diff(np.sort(kpts[:, c])))

        # Determine number of k-points as inverse of distance between kpoints
        if delta > 1e-8:
            size[c] = int(round(1.0 / delta))
        else:
            size[c] = 1

    if size.prod() == len(kpts):
        kpts0 = monkhorst_pack(size)
        offsets = kpts - kpts0

        # All offsets must be identical:
        if (np.ptp(offsets, axis=0) < 1e-9).all():
            return size, offsets[0].copy()

    raise ValueError('Not an ASE-style Monkhorst-Pack grid!')


def mindistance2monkhorstpack(atoms, *,
                              min_distance,
                              maxperdim=16,
                              even=True):
    """Find a Monkhorst-Pack grid (nx, ny, nz) with lowest number of
       k-points in the *reducible* Brillouin zone, which still satisfying
       a given minimum distance (`min_distance`) condition in real space
       (nx, ny, nz)-supercell.

       Compared to ase.calculators.calculator kptdensity2monkhorstpack
       routine, this metric is based on a physical quantity (real space
       distance), and it doesn't depend on non-physical quantities, such as
       the cell vectors, since basis vectors can be always transformed
       with integer determinant one matrices. In other words, it is
       invariant to particular choice of cell representations.

       On orthogonal cells, min_distance = 2 * np.pi * kptdensity.
    """
    return _mindistance2monkhorstpack(atoms.cell, atoms.pbc,
                                      min_distance, maxperdim, even)


def _mindistance2monkhorstpack(cell, pbc_c, min_distance, maxperdim, even):
    from ase import Atoms
    from ase.neighborlist import NeighborList

    step = 2 if even else 1
    nl = NeighborList([min_distance / 2], skin=0.0,
                      self_interaction=False, bothways=False)

    def check(nkpts_c):
        nl.update(Atoms('H', cell=cell @ np.diag(nkpts_c), pbc=pbc_c))
        return len(nl.get_neighbors(0)[1]) == 0

    def generate_mpgrids():
        ranges = [range(step, maxperdim + 1, step)
                  if pbc else range(1, 2) for pbc in pbc_c]
        nkpts_nc = np.column_stack([*map(np.ravel, np.meshgrid(*ranges))])
        yield from sorted(nkpts_nc, key=np.prod)

    try:
        return next(filter(check, generate_mpgrids()))
    except StopIteration:
        raise ValueError('Could not find a proper k-point grid for the system.'
                         ' Try running with a larger maxperdim.')


def get_monkhorst_shape(kpts):
    warnings.warn('Use get_monkhorst_pack_size_and_offset()[0] instead.')
    return get_monkhorst_pack_size_and_offset(kpts)[0]


def kpoint_convert(cell_cv, skpts_kc=None, ckpts_kv=None):
    """Convert k-points between scaled and cartesian coordinates.

    Given the atomic unit cell, and either the scaled or cartesian k-point
    coordinates, the other is determined.

    The k-point arrays can be either a single point, or a list of points,
    i.e. the dimension k can be empty or multidimensional.
    """
    if ckpts_kv is None:
        icell_cv = 2 * np.pi * np.linalg.pinv(cell_cv).T
        return np.dot(skpts_kc, icell_cv)
    elif skpts_kc is None:
        return np.dot(ckpts_kv, cell_cv.T) / (2 * np.pi)
    else:
        raise KeyError('Either scaled or cartesian coordinates must be given.')


def parse_path_string(s):
    """Parse compact string representation of BZ path.

    A path string can have several non-connected sections separated by
    commas. The return value is a list of sections where each section is a
    list of labels.

    Examples:

    >>> parse_path_string('GX')
    [['G', 'X']]
    >>> parse_path_string('GX,M1A')
    [['G', 'X'], ['M1', 'A']]
    """
    paths = []
    for path in s.split(','):
        names = [name
                 for name in re.split(r'([A-Z][a-z0-9]*)', path)
                 if name]
        paths.append(names)
    return paths


def resolve_kpt_path_string(path, special_points):
    paths = parse_path_string(path)
    coords = [np.array([special_points[sym] for sym in subpath]).reshape(-1, 3)
              for subpath in paths]
    return paths, coords


def resolve_custom_points(pathspec, special_points, eps):
    """Resolve a path specification into a string.

    The path specification is a list path segments, each segment being a kpoint
    label or kpoint coordinate, or a single such segment.

    Return a string representing the same path.  Generic kpoint labels
    are generated dynamically as necessary, updating the special_point
    dictionary if necessary.  The tolerance eps is used to see whether
    coordinates are close enough to a special point to deserve being
    labelled as such."""
    # This should really run on Cartesian coordinates but we'll probably
    # be lazy and call it on scaled ones.

    # We may add new points below so take a copy of the input:
    special_points = dict(special_points)

    if len(pathspec) == 0:
        return '', special_points

    if isinstance(pathspec, str):
        pathspec = parse_path_string(pathspec)

    def looks_like_single_kpoint(obj):
        if isinstance(obj, str):
            return True
        try:
            arr = np.asarray(obj, float)
        except ValueError:
            return False
        else:
            return arr.shape == (3,)

    # We accept inputs that are either
    #  1) string notation
    #  2) list of kpoints (each either a string or three floats)
    #  3) list of list of kpoints; each toplevel list is a contiguous subpath
    # Here we detect form 2 and normalize to form 3:
    for thing in pathspec:
        if looks_like_single_kpoint(thing):
            pathspec = [pathspec]
            break

    def name_generator():
        counter = 0
        while True:
            name = f'Kpt{counter}'
            yield name
            counter += 1

    custom_names = name_generator()

    labelseq = []
    for subpath in pathspec:
        for kpt in subpath:
            if isinstance(kpt, str):
                if kpt not in special_points:
                    raise KeyError('No kpoint "{}" among "{}"'
                                   .format(kpt,
                                           ''.join(special_points)))
                labelseq.append(kpt)
                continue

            kpt = np.asarray(kpt, float)
            if kpt.shape != (3,):
                raise ValueError(f'Not a valid kpoint: {kpt}')

            for key, val in special_points.items():
                if np.abs(kpt - val).max() < eps:
                    labelseq.append(key)
                    break  # Already present
            else:
                # New special point - search for name we haven't used yet:
                name = next(custom_names)
                while name in special_points:
                    name = next(custom_names)
                special_points[name] = kpt
                labelseq.append(name)
        labelseq.append(',')

    last = labelseq.pop()
    assert last == ','
    return ''.join(labelseq), special_points


def normalize_special_points(special_points):
    dct = {}
    for name, value in special_points.items():
        if not isinstance(name, str):
            raise TypeError('Expected name to be a string')
        if np.shape(value) != (3,):
            raise ValueError('Expected 3 kpoint coordinates')
        dct[name] = np.asarray(value, float)
    return dct


@jsonable('bandpath')
class BandPath:
    """Represents a Brillouin zone path or bandpath.

    A band path has a unit cell, a path specification, special points,
    and interpolated k-points.  Band paths are typically created
    indirectly using the :class:`~ase.geometry.Cell` or
    :class:`~ase.lattice.BravaisLattice` classes:

    >>> from ase.lattice import CUB
    >>> path = CUB(3).bandpath()
    >>> path
    BandPath(path='GXMGRX,MR', cell=[3x3], special_points={GMRX}, kpts=[40x3])

    Band paths support JSON I/O:

    >>> from ase.io.jsonio import read_json
    >>> path.write('mybandpath.json')
    >>> read_json('mybandpath.json')
    BandPath(path='GXMGRX,MR', cell=[3x3], special_points={GMRX}, kpts=[40x3])

    """

    def __init__(self, cell, kpts=None,
                 special_points=None, path=None):
        if kpts is None:
            kpts = np.empty((0, 3))

        if special_points is None:
            special_points = {}
        else:
            special_points = normalize_special_points(special_points)

        if path is None:
            path = ''

        cell = Cell(cell)
        self._cell = cell
        kpts = np.asarray(kpts)
        assert kpts.ndim == 2 and kpts.shape[1] == 3 and kpts.dtype == float
        self._icell = self.cell.reciprocal()
        self._kpts = kpts
        self._special_points = special_points
        if not isinstance(path, str):
            raise TypeError(f'path must be a string; was {path!r}')
        self._path = path

    @property
    def cell(self) -> Cell:
        """The :class:`~ase.cell.Cell` of this BandPath."""
        return self._cell

    @property
    def icell(self) -> Cell:
        """Reciprocal cell of this BandPath as a :class:`~ase.cell.Cell`."""
        return self._icell

    @property
    def kpts(self) -> np.ndarray:
        """The kpoints of this BandPath as an array of shape (nkpts, 3).

        The kpoints are given in units of the reciprocal cell."""
        return self._kpts

    @property
    def special_points(self) -> Dict[str, np.ndarray]:
        """Special points of this BandPath as a dictionary.

        The dictionary maps names (such as `'G'`) to kpoint coordinates
        in units of the reciprocal cell as a 3-element numpy array.

        It's unwise to edit this dictionary directly.  If you need that,
        consider deepcopying it."""
        return self._special_points

    @property
    def path(self) -> str:
        """The string specification of this band path.

        This is a specification of the form `'GXWKGLUWLK,UX'`.

        Comma marks a discontinuous jump: K is not connected to U."""
        return self._path

    def transform(self, op: np.ndarray) -> 'BandPath':
        """Apply 3x3 matrix to this BandPath and return new BandPath.

        This is useful for converting the band path to another cell.
        The operation will typically be a permutation/flipping
        established by a function such as Niggli reduction."""
        # XXX acceptable operations are probably only those
        # who come from Niggli reductions (permutations etc.)
        #
        # We should insert a check.
        # I wonder which operations are valid?  They won't be valid
        # if they change lengths, volume etc.
        special_points = {
            name: value @ op for name, value in self.special_points.items()
        }
        return BandPath(op.T @ self.cell, kpts=self.kpts @ op,
                        special_points=special_points,
                        path=self.path)

    def todict(self):
        return {'kpts': self.kpts,
                'special_points': self.special_points,
                'labelseq': self.path,
                'cell': self.cell}

    def interpolate(
            self,
            path: str = None,
            npoints: int = None,
            special_points: Dict[str, np.ndarray] = None,
            density: float = None,
    ) -> 'BandPath':
        """Create new bandpath, (re-)interpolating kpoints from this one."""
        if path is None:
            path = self.path

        if special_points is None:
            special_points = self.special_points

        _pathnames, pathcoords = resolve_kpt_path_string(path, special_points)
        kpts, _x, _X = paths2kpts(pathcoords, self.cell, npoints, density)
        return BandPath(self.cell, kpts, path=path,
                        special_points=special_points)

    def _scale(self, coords):
        return np.dot(coords, self.icell)

    def __repr__(self):
        return ('{}(path={}, cell=[3x3], special_points={{{}}}, kpts=[{}x3])'
                .format(self.__class__.__name__,
                        repr(self.path),
                        ''.join(sorted(self.special_points)),
                        len(self.kpts)))

    def cartesian_kpts(self) -> np.ndarray:
        """Get Cartesian kpoints from this bandpath."""
        return self._scale(self.kpts)

    def __iter__(self):
        """XXX Compatibility hack for bandpath() function.

        bandpath() now returns a BandPath object, which is a Good
        Thing.  However it used to return a tuple of (kpts, x_axis,
        special_x_coords), and people would use tuple unpacking for
        those.

        This function makes tuple unpacking work in the same way.
        It will be removed in the future.

        """
        import warnings
        warnings.warn('Please do not use (kpts, x, X) = bandpath(...).  '
                      'Use path = bandpath(...) and then kpts = path.kpts and '
                      '(x, X, labels) = path.get_linear_kpoint_axis().')
        yield self.kpts

        x, xspecial, _ = self.get_linear_kpoint_axis()
        yield x
        yield xspecial

    def __getitem__(self, index):
        # Temp compatibility stuff, see __iter__
        return tuple(self)[index]

    def get_linear_kpoint_axis(self, eps=1e-5):
        """Define x axis suitable for plotting a band structure.

        See :func:`ase.dft.kpoints.labels_from_kpts`."""

        index2name = self._find_special_point_indices(eps)
        indices = sorted(index2name)
        labels = [index2name[index] for index in indices]
        xcoords, special_xcoords = indices_to_axis_coords(
            indices, self.kpts, self.cell)
        return xcoords, special_xcoords, labels

    def _find_special_point_indices(self, eps):
        """Find indices of kpoints which are close to special points.

        The result is returned as a dictionary mapping indices to labels."""
        # XXX must work in Cartesian coordinates for comparison to eps
        # to fully make sense!
        index2name = {}
        nkpts = len(self.kpts)

        for name, kpt in self.special_points.items():
            displacements = self.kpts - kpt[np.newaxis, :]
            distances = np.linalg.norm(displacements, axis=1)
            args = np.argwhere(distances < eps)
            for arg in args.flat:
                dist = distances[arg]
                # Check if an adjacent point exists and is even closer:
                neighbours = distances[max(arg - 1, 0):min(arg + 1, nkpts - 1)]
                if not any(neighbours < dist):
                    index2name[arg] = name

        return index2name

    def plot(self, **plotkwargs):
        """Visualize this bandpath.

        Plots the irreducible Brillouin zone and this bandpath."""
        import ase.dft.bz as bz

        # We previously had a "dimension=3" argument which is now unused.
        plotkwargs.pop('dimension', None)

        special_points = self.special_points
        labelseq, coords = resolve_kpt_path_string(self.path,
                                                   special_points)

        paths = []
        points_already_plotted = set()
        for subpath_labels, subpath_coords in zip(labelseq, coords):
            subpath_coords = np.array(subpath_coords)
            points_already_plotted.update(subpath_labels)
            paths.append((subpath_labels, self._scale(subpath_coords)))

        # Add each special point as a single-point subpath if they were
        # not plotted already:
        for label, point in special_points.items():
            if label not in points_already_plotted:
                paths.append(([label], [self._scale(point)]))

        kw = {'vectors': True,
              'pointstyle': {'marker': '.'}}

        kw.update(plotkwargs)
        return bz.bz_plot(self.cell, paths=paths,
                          points=self.cartesian_kpts(),
                          **kw)

    def free_electron_band_structure(
            self, **kwargs
    ) -> 'ase.spectrum.band_structure.BandStructure':
        """Return band structure of free electrons for this bandpath.

        Keyword arguments are passed to
        :class:`~ase.calculators.test.FreeElectrons`.

        This is for mostly testing and visualization."""
        from ase import Atoms
        from ase.calculators.test import FreeElectrons
        from ase.spectrum.band_structure import calculate_band_structure
        atoms = Atoms(cell=self.cell, pbc=True)
        atoms.calc = FreeElectrons(**kwargs)
        bs = calculate_band_structure(atoms, path=self)
        return bs


def bandpath(path, cell, npoints=None, density=None, special_points=None,
             eps=2e-4):
    """Make a list of kpoints defining the path between the given points.

    path: list or str
        Can be:

        * a string that parse_path_string() understands: 'GXL'
        * a list of BZ points: [(0, 0, 0), (0.5, 0, 0)]
        * or several lists of BZ points if the the path is not continuous.
    cell: 3x3
        Unit cell of the atoms.
    npoints: int
        Length of the output kpts list. If too small, at least the beginning
        and ending point of each path segment will be used. If None (default),
        it will be calculated using the supplied density or a default one.
    density: float
        k-points per 1/A on the output kpts list. If npoints is None,
        the number of k-points in the output list will be:
        npoints = density * path total length (in Angstroms).
        If density is None (default), use 5 k-points per A⁻¹.
        If the calculated npoints value is less than 50, a minimum value of 50
        will be used.
    special_points: dict or None
        Dictionary mapping names to special points.  If None, the special
        points will be derived from the cell.
    eps: float
        Precision used to identify Bravais lattice, deducing special points.

    You may define npoints or density but not both.

    Return a :class:`~ase.dft.kpoints.BandPath` object."""

    cell = Cell.ascell(cell)
    return cell.bandpath(path, npoints=npoints, density=density,
                         special_points=special_points, eps=eps)


DEFAULT_KPTS_DENSITY = 5    # points per 1/Angstrom


def paths2kpts(paths, cell, npoints=None, density=None):
    if npoints is not None and density is not None:
        raise ValueError('You may define npoints or density, but not both.')
    points = np.concatenate(paths)
    dists = points[1:] - points[:-1]
    lengths = [np.linalg.norm(d) for d in kpoint_convert(cell, skpts_kc=dists)]

    i = 0
    for path in paths[:-1]:
        i += len(path)
        lengths[i - 1] = 0

    length = sum(lengths)

    if npoints is None:
        if density is None:
            density = DEFAULT_KPTS_DENSITY
        # Set npoints using the length of the path
        npoints = int(round(length * density))

    kpts = []
    x0 = 0
    x = []
    X = [0]
    for P, d, L in zip(points[:-1], dists, lengths):
        diff = length - x0
        if abs(diff) < 1e-6:
            n = 0
        else:
            n = max(2, int(round(L * (npoints - len(x)) / diff)))

        for t in np.linspace(0, 1, n)[:-1]:
            kpts.append(P + t * d)
            x.append(x0 + t * L)
        x0 += L
        X.append(x0)
    if len(points):
        kpts.append(points[-1])
        x.append(x0)

    if len(kpts) == 0:
        kpts = np.empty((0, 3))

    return np.array(kpts), np.array(x), np.array(X)


get_bandpath = bandpath  # old name


def find_bandpath_kinks(cell, kpts, eps=1e-5):
    """Find indices of those kpoints that are not interior to a line segment."""
    # XXX Should use the Cartesian kpoints.
    # Else comparison to eps will be anisotropic and hence arbitrary.
    diffs = kpts[1:] - kpts[:-1]
    kinks = abs(diffs[1:] - diffs[:-1]).sum(1) > eps
    N = len(kpts)
    indices = []
    if N > 0:
        indices.append(0)
        indices.extend(np.arange(1, N - 1)[kinks])
        indices.append(N - 1)
    return indices


def labels_from_kpts(kpts, cell, eps=1e-5, special_points=None):
    """Get an x-axis to be used when plotting a band structure.

    The first of the returned lists can be used as a x-axis when plotting
    the band structure. The second list can be used as xticks, and the third
    as xticklabels.

    Parameters:

    kpts: list
        List of scaled k-points.

    cell: list
        Unit cell of the atomic structure.

    Returns:

    Three arrays; the first is a list of cumulative distances between k-points,
    the second is x coordinates of the special points,
    the third is the special points as strings.
    """

    if special_points is None:
        special_points = get_special_points(cell)
    points = np.asarray(kpts)
    # XXX Due to this mechanism, we are blind to special points
    # that lie on straight segments such as [K, G, -K].
    indices = find_bandpath_kinks(cell, kpts, eps=1e-5)

    labels = []
    for kpt in points[indices]:
        for label, k in special_points.items():
            if abs(kpt - k).sum() < eps:
                break
        else:
            # No exact match.  Try modulus 1:
            for label, k in special_points.items():
                if abs((kpt - k) % 1).sum() < eps:
                    break
            else:
                label = '?'
        labels.append(label)

    xcoords, ixcoords = indices_to_axis_coords(indices, points, cell)
    return xcoords, ixcoords, labels


def indices_to_axis_coords(indices, points, cell):
    jump = False  # marks a discontinuity in the path
    xcoords = [0]
    for i1, i2 in zip(indices[:-1], indices[1:]):
        if not jump and i1 + 1 == i2:
            length = 0
            jump = True  # we don't want two jumps in a row
        else:
            diff = points[i2] - points[i1]
            length = np.linalg.norm(kpoint_convert(cell, skpts_kc=diff))
            jump = False
        xcoords.extend(np.linspace(0, length, i2 - i1 + 1)[1:] + xcoords[-1])

    xcoords = np.array(xcoords)
    return xcoords, xcoords[indices]


special_paths = {
    'cubic': 'GXMGRX,MR',
    'fcc': 'GXWKGLUWLK,UX',
    'bcc': 'GHNGPH,PN',
    'tetragonal': 'GXMGZRAZXR,MA',
    'orthorhombic': 'GXSYGZURTZ,YT,UX,SR',
    'hexagonal': 'GMKGALHA,LM,KH',
    'monoclinic': 'GYHCEM1AXH1,MDZ,YD',
    'rhombohedral type 1': 'GLB1,BZGX,QFP1Z,LP',
    'rhombohedral type 2': 'GPZQGFP1Q1LZ'}


def get_special_points(cell, lattice=None, eps=2e-4):
    """Return dict of special points.

    The definitions are from a paper by Wahyu Setyawana and Stefano
    Curtarolo::

        https://doi.org/10.1016/j.commatsci.2010.05.010

    cell: 3x3 ndarray
        Unit cell.
    lattice: str
        Optionally check that the cell is one of the following: cubic, fcc,
        bcc, orthorhombic, tetragonal, hexagonal or monoclinic.
    eps: float
        Tolerance for cell-check.
    """

    if isinstance(cell, str):
        warnings.warn('Please call this function with cell as the first '
                      'argument')
        lattice, cell = cell, lattice

    cell = Cell.ascell(cell)
    # We create the bandpath because we want to transform the kpoints too,
    # from the canonical cell to the given one.
    #
    # Note that this function is missing a tolerance, epsilon.
    path = cell.bandpath(npoints=0)
    return path.special_points


def monkhorst_pack_interpolate(path, values, icell, bz2ibz,
                               size, offset=(0, 0, 0), pad_width=2):
    """Interpolate values from Monkhorst-Pack sampling.

    `monkhorst_pack_interpolate` takes a set of `values`, for example
    eigenvalues, that are resolved on a Monkhorst Pack k-point grid given by
    `size` and `offset` and interpolates the values onto the k-points
    in `path`.

    Note
    ----
    For the interpolation to work, path has to lie inside the domain
    that is spanned by the MP kpoint grid given by size and offset.

    To try to ensure this we expand the domain slightly by adding additional
    entries along the edges and sides of the domain with values determined by
    wrapping the values to the opposite side of the domain. In this way we
    assume that the function to be interpolated is a periodic function in
    k-space. The padding width is determined by the `pad_width` parameter.

    Parameters
    ----------
    path: (nk, 3) array-like
        Desired path in units of reciprocal lattice vectors.
    values: (nibz, ...) array-like
        Values on Monkhorst-Pack grid.
    icell: (3, 3) array-like
        Reciprocal lattice vectors.
    bz2ibz: (nbz,) array-like of int
        Map from nbz points in BZ to nibz reduced points in IBZ.
    size: (3,) array-like of int
        Size of Monkhorst-Pack grid.
    offset: (3,) array-like
        Offset of Monkhorst-Pack grid.
    pad_width: int
        Padding width to aid interpolation

    Returns
    -------
    (nbz,) array-like
        *values* interpolated to *path*.
    """
    from scipy.interpolate import LinearNDInterpolator

    path = (np.asarray(path) + 0.5) % 1 - 0.5
    path = np.dot(path, icell)

    # Fold out values from IBZ to BZ:
    v = np.asarray(values)[bz2ibz]
    v = v.reshape(tuple(size) + v.shape[1:])

    # Create padded Monkhorst-Pack grid:
    size = np.asarray(size)
    i = (np.indices(size + 2 * pad_width)
         .transpose((1, 2, 3, 0)).reshape((-1, 3)))
    k = (i - pad_width + 0.5) / size - 0.5 + offset
    k = np.dot(k, icell)

    # Fill in boundary values:
    V = np.pad(v, [(pad_width, pad_width)] * 3 +
               [(0, 0)] * (v.ndim - 3), mode="wrap")

    interpolate = LinearNDInterpolator(k, V.reshape((-1,) + V.shape[3:]))
    interpolated_points = interpolate(path)

    # NaN values indicate points outside interpolation domain, if fail
    # try increasing padding
    assert not np.isnan(interpolated_points).any(), \
        "Points outside interpolation domain. Try increasing pad_width."

    return interpolated_points


# ChadiCohen k point grids. The k point grids are given in units of the
# reciprocal unit cell. The variables are named after the following
# convention: cc+'<Nkpoints>'+_+'shape'. For example an 18 k point
# sq(3)xsq(3) is named 'cc18_sq3xsq3'.

cc6_1x1 = np.array([
    1, 1, 0, 1, 0, 0, 0, -1, 0, -1, -1, 0, -1, 0, 0,
    0, 1, 0]).reshape((6, 3)) / 3.0

cc12_2x3 = np.array([
    3, 4, 0, 3, 10, 0, 6, 8, 0, 3, -2, 0, 6, -4, 0,
    6, 2, 0, -3, 8, 0, -3, 2, 0, -3, -4, 0, -6, 4, 0, -6, -2, 0, -6,
    -8, 0]).reshape((12, 3)) / 18.0

cc18_sq3xsq3 = np.array([
    2, 2, 0, 4, 4, 0, 8, 2, 0, 4, -2, 0, 8, -4,
    0, 10, -2, 0, 10, -8, 0, 8, -10, 0, 2, -10, 0, 4, -8, 0, -2, -8,
    0, 2, -4, 0, -4, -4, 0, -2, -2, 0, -4, 2, 0, -2, 4, 0, -8, 4, 0,
    -4, 8, 0]).reshape((18, 3)) / 18.0

cc18_1x1 = np.array([
    2, 4, 0, 2, 10, 0, 4, 8, 0, 8, 4, 0, 8, 10, 0,
    10, 8, 0, 2, -2, 0, 4, -4, 0, 4, 2, 0, -2, 8, 0, -2, 2, 0, -2, -4,
    0, -4, 4, 0, -4, -2, 0, -4, -8, 0, -8, 2, 0, -8, -4, 0, -10, -2,
    0]).reshape((18, 3)) / 18.0

cc54_sq3xsq3 = np.array([
    4, -10, 0, 6, -10, 0, 0, -8, 0, 2, -8, 0, 6,
    -8, 0, 8, -8, 0, -4, -6, 0, -2, -6, 0, 2, -6, 0, 4, -6, 0, 8, -6,
    0, 10, -6, 0, -6, -4, 0, -2, -4, 0, 0, -4, 0, 4, -4, 0, 6, -4, 0,
    10, -4, 0, -6, -2, 0, -4, -2, 0, 0, -2, 0, 2, -2, 0, 6, -2, 0, 8,
    -2, 0, -8, 0, 0, -4, 0, 0, -2, 0, 0, 2, 0, 0, 4, 0, 0, 8, 0, 0,
    -8, 2, 0, -6, 2, 0, -2, 2, 0, 0, 2, 0, 4, 2, 0, 6, 2, 0, -10, 4,
    0, -6, 4, 0, -4, 4, 0, 0, 4, 0, 2, 4, 0, 6, 4, 0, -10, 6, 0, -8,
    6, 0, -4, 6, 0, -2, 6, 0, 2, 6, 0, 4, 6, 0, -8, 8, 0, -6, 8, 0,
    -2, 8, 0, 0, 8, 0, -6, 10, 0, -4, 10, 0]).reshape((54, 3)) / 18.0

cc54_1x1 = np.array([
    2, 2, 0, 4, 4, 0, 8, 8, 0, 6, 8, 0, 4, 6, 0, 6,
    10, 0, 4, 10, 0, 2, 6, 0, 2, 8, 0, 0, 2, 0, 0, 4, 0, 0, 8, 0, -2,
    6, 0, -2, 4, 0, -4, 6, 0, -6, 4, 0, -4, 2, 0, -6, 2, 0, -2, 0, 0,
    -4, 0, 0, -8, 0, 0, -8, -2, 0, -6, -2, 0, -10, -4, 0, -10, -6, 0,
    -6, -4, 0, -8, -6, 0, -2, -2, 0, -4, -4, 0, -8, -8, 0, 4, -2, 0,
    6, -2, 0, 6, -4, 0, 2, 0, 0, 4, 0, 0, 6, 2, 0, 6, 4, 0, 8, 6, 0,
    8, 0, 0, 8, 2, 0, 10, 4, 0, 10, 6, 0, 2, -4, 0, 2, -6, 0, 4, -6,
    0, 0, -2, 0, 0, -4, 0, -2, -6, 0, -4, -6, 0, -6, -8, 0, 0, -8, 0,
    -2, -8, 0, -4, -10, 0, -6, -10, 0]).reshape((54, 3)) / 18.0

cc162_sq3xsq3 = np.array([
    -8, 16, 0, -10, 14, 0, -7, 14, 0, -4, 14,
    0, -11, 13, 0, -8, 13, 0, -5, 13, 0, -2, 13, 0, -13, 11, 0, -10,
    11, 0, -7, 11, 0, -4, 11, 0, -1, 11, 0, 2, 11, 0, -14, 10, 0, -11,
    10, 0, -8, 10, 0, -5, 10, 0, -2, 10, 0, 1, 10, 0, 4, 10, 0, -16,
    8, 0, -13, 8, 0, -10, 8, 0, -7, 8, 0, -4, 8, 0, -1, 8, 0, 2, 8, 0,
    5, 8, 0, 8, 8, 0, -14, 7, 0, -11, 7, 0, -8, 7, 0, -5, 7, 0, -2, 7,
    0, 1, 7, 0, 4, 7, 0, 7, 7, 0, 10, 7, 0, -13, 5, 0, -10, 5, 0, -7,
    5, 0, -4, 5, 0, -1, 5, 0, 2, 5, 0, 5, 5, 0, 8, 5, 0, 11, 5, 0,
    -14, 4, 0, -11, 4, 0, -8, 4, 0, -5, 4, 0, -2, 4, 0, 1, 4, 0, 4, 4,
    0, 7, 4, 0, 10, 4, 0, -13, 2, 0, -10, 2, 0, -7, 2, 0, -4, 2, 0,
    -1, 2, 0, 2, 2, 0, 5, 2, 0, 8, 2, 0, 11, 2, 0, -11, 1, 0, -8, 1,
    0, -5, 1, 0, -2, 1, 0, 1, 1, 0, 4, 1, 0, 7, 1, 0, 10, 1, 0, 13, 1,
    0, -10, -1, 0, -7, -1, 0, -4, -1, 0, -1, -1, 0, 2, -1, 0, 5, -1,
    0, 8, -1, 0, 11, -1, 0, 14, -1, 0, -11, -2, 0, -8, -2, 0, -5, -2,
    0, -2, -2, 0, 1, -2, 0, 4, -2, 0, 7, -2, 0, 10, -2, 0, 13, -2, 0,
    -10, -4, 0, -7, -4, 0, -4, -4, 0, -1, -4, 0, 2, -4, 0, 5, -4, 0,
    8, -4, 0, 11, -4, 0, 14, -4, 0, -8, -5, 0, -5, -5, 0, -2, -5, 0,
    1, -5, 0, 4, -5, 0, 7, -5, 0, 10, -5, 0, 13, -5, 0, 16, -5, 0, -7,
    -7, 0, -4, -7, 0, -1, -7, 0, 2, -7, 0, 5, -7, 0, 8, -7, 0, 11, -7,
    0, 14, -7, 0, 17, -7, 0, -8, -8, 0, -5, -8, 0, -2, -8, 0, 1, -8,
    0, 4, -8, 0, 7, -8, 0, 10, -8, 0, 13, -8, 0, 16, -8, 0, -7, -10,
    0, -4, -10, 0, -1, -10, 0, 2, -10, 0, 5, -10, 0, 8, -10, 0, 11,
    -10, 0, 14, -10, 0, 17, -10, 0, -5, -11, 0, -2, -11, 0, 1, -11, 0,
    4, -11, 0, 7, -11, 0, 10, -11, 0, 13, -11, 0, 16, -11, 0, -1, -13,
    0, 2, -13, 0, 5, -13, 0, 8, -13, 0, 11, -13, 0, 14, -13, 0, 1,
    -14, 0, 4, -14, 0, 7, -14, 0, 10, -14, 0, 13, -14, 0, 5, -16, 0,
    8, -16, 0, 11, -16, 0, 7, -17, 0, 10, -17, 0]).reshape((162, 3)) / 27.0

cc162_1x1 = np.array([
    -8, -16, 0, -10, -14, 0, -7, -14, 0, -4, -14,
    0, -11, -13, 0, -8, -13, 0, -5, -13, 0, -2, -13, 0, -13, -11, 0,
    -10, -11, 0, -7, -11, 0, -4, -11, 0, -1, -11, 0, 2, -11, 0, -14,
    -10, 0, -11, -10, 0, -8, -10, 0, -5, -10, 0, -2, -10, 0, 1, -10,
    0, 4, -10, 0, -16, -8, 0, -13, -8, 0, -10, -8, 0, -7, -8, 0, -4,
    -8, 0, -1, -8, 0, 2, -8, 0, 5, -8, 0, 8, -8, 0, -14, -7, 0, -11,
    -7, 0, -8, -7, 0, -5, -7, 0, -2, -7, 0, 1, -7, 0, 4, -7, 0, 7, -7,
    0, 10, -7, 0, -13, -5, 0, -10, -5, 0, -7, -5, 0, -4, -5, 0, -1,
    -5, 0, 2, -5, 0, 5, -5, 0, 8, -5, 0, 11, -5, 0, -14, -4, 0, -11,
    -4, 0, -8, -4, 0, -5, -4, 0, -2, -4, 0, 1, -4, 0, 4, -4, 0, 7, -4,
    0, 10, -4, 0, -13, -2, 0, -10, -2, 0, -7, -2, 0, -4, -2, 0, -1,
    -2, 0, 2, -2, 0, 5, -2, 0, 8, -2, 0, 11, -2, 0, -11, -1, 0, -8,
    -1, 0, -5, -1, 0, -2, -1, 0, 1, -1, 0, 4, -1, 0, 7, -1, 0, 10, -1,
    0, 13, -1, 0, -10, 1, 0, -7, 1, 0, -4, 1, 0, -1, 1, 0, 2, 1, 0, 5,
    1, 0, 8, 1, 0, 11, 1, 0, 14, 1, 0, -11, 2, 0, -8, 2, 0, -5, 2, 0,
    -2, 2, 0, 1, 2, 0, 4, 2, 0, 7, 2, 0, 10, 2, 0, 13, 2, 0, -10, 4,
    0, -7, 4, 0, -4, 4, 0, -1, 4, 0, 2, 4, 0, 5, 4, 0, 8, 4, 0, 11, 4,
    0, 14, 4, 0, -8, 5, 0, -5, 5, 0, -2, 5, 0, 1, 5, 0, 4, 5, 0, 7, 5,
    0, 10, 5, 0, 13, 5, 0, 16, 5, 0, -7, 7, 0, -4, 7, 0, -1, 7, 0, 2,
    7, 0, 5, 7, 0, 8, 7, 0, 11, 7, 0, 14, 7, 0, 17, 7, 0, -8, 8, 0,
    -5, 8, 0, -2, 8, 0, 1, 8, 0, 4, 8, 0, 7, 8, 0, 10, 8, 0, 13, 8, 0,
    16, 8, 0, -7, 10, 0, -4, 10, 0, -1, 10, 0, 2, 10, 0, 5, 10, 0, 8,
    10, 0, 11, 10, 0, 14, 10, 0, 17, 10, 0, -5, 11, 0, -2, 11, 0, 1,
    11, 0, 4, 11, 0, 7, 11, 0, 10, 11, 0, 13, 11, 0, 16, 11, 0, -1,
    13, 0, 2, 13, 0, 5, 13, 0, 8, 13, 0, 11, 13, 0, 14, 13, 0, 1, 14,
    0, 4, 14, 0, 7, 14, 0, 10, 14, 0, 13, 14, 0, 5, 16, 0, 8, 16, 0,
    11, 16, 0, 7, 17, 0, 10, 17, 0]).reshape((162, 3)) / 27.0


# The following is a list of the critical points in the 1st Brillouin zone
# for some typical crystal structures following the conventions of Setyawan
# and Curtarolo [https://doi.org/10.1016/j.commatsci.2010.05.010].
#
# In units of the reciprocal basis vectors.
#
# See http://en.wikipedia.org/wiki/Brillouin_zone
sc_special_points = {
    'cubic': {'G': [0, 0, 0],
              'M': [1 / 2, 1 / 2, 0],
              'R': [1 / 2, 1 / 2, 1 / 2],
              'X': [0, 1 / 2, 0]},
    'fcc': {'G': [0, 0, 0],
            'K': [3 / 8, 3 / 8, 3 / 4],
            'L': [1 / 2, 1 / 2, 1 / 2],
            'U': [5 / 8, 1 / 4, 5 / 8],
            'W': [1 / 2, 1 / 4, 3 / 4],
            'X': [1 / 2, 0, 1 / 2]},
    'bcc': {'G': [0, 0, 0],
            'H': [1 / 2, -1 / 2, 1 / 2],
            'P': [1 / 4, 1 / 4, 1 / 4],
            'N': [0, 0, 1 / 2]},
    'tetragonal': {'G': [0, 0, 0],
                   'A': [1 / 2, 1 / 2, 1 / 2],
                   'M': [1 / 2, 1 / 2, 0],
                   'R': [0, 1 / 2, 1 / 2],
                   'X': [0, 1 / 2, 0],
                   'Z': [0, 0, 1 / 2]},
    'orthorhombic': {'G': [0, 0, 0],
                     'R': [1 / 2, 1 / 2, 1 / 2],
                     'S': [1 / 2, 1 / 2, 0],
                     'T': [0, 1 / 2, 1 / 2],
                     'U': [1 / 2, 0, 1 / 2],
                     'X': [1 / 2, 0, 0],
                     'Y': [0, 1 / 2, 0],
                     'Z': [0, 0, 1 / 2]},
    'hexagonal': {'G': [0, 0, 0],
                  'A': [0, 0, 1 / 2],
                  'H': [1 / 3, 1 / 3, 1 / 2],
                  'K': [1 / 3, 1 / 3, 0],
                  'L': [1 / 2, 0, 1 / 2],
                  'M': [1 / 2, 0, 0]}}


# Old version of dictionary kept for backwards compatibility.
# Not for ordinary use.
ibz_points = {'cubic': {'Gamma': [0, 0, 0],
                        'X': [0, 0 / 2, 1 / 2],
                        'R': [1 / 2, 1 / 2, 1 / 2],
                        'M': [0 / 2, 1 / 2, 1 / 2]},
              'fcc': {'Gamma': [0, 0, 0],
                      'X': [1 / 2, 0, 1 / 2],
                      'W': [1 / 2, 1 / 4, 3 / 4],
                      'K': [3 / 8, 3 / 8, 3 / 4],
                      'U': [5 / 8, 1 / 4, 5 / 8],
                      'L': [1 / 2, 1 / 2, 1 / 2]},
              'bcc': {'Gamma': [0, 0, 0],
                      'H': [1 / 2, -1 / 2, 1 / 2],
                      'N': [0, 0, 1 / 2],
                      'P': [1 / 4, 1 / 4, 1 / 4]},
              'hexagonal': {'Gamma': [0, 0, 0],
                            'M': [0, 1 / 2, 0],
                            'K': [-1 / 3, 1 / 3, 0],
                            'A': [0, 0, 1 / 2],
                            'L': [0, 1 / 2, 1 / 2],
                            'H': [-1 / 3, 1 / 3, 1 / 2]},
              'tetragonal': {'Gamma': [0, 0, 0],
                             'X': [1 / 2, 0, 0],
                             'M': [1 / 2, 1 / 2, 0],
                             'Z': [0, 0, 1 / 2],
                             'R': [1 / 2, 0, 1 / 2],
                             'A': [1 / 2, 1 / 2, 1 / 2]},
              'orthorhombic': {'Gamma': [0, 0, 0],
                               'R': [1 / 2, 1 / 2, 1 / 2],
                               'S': [1 / 2, 1 / 2, 0],
                               'T': [0, 1 / 2, 1 / 2],
                               'U': [1 / 2, 0, 1 / 2],
                               'X': [1 / 2, 0, 0],
                               'Y': [0, 1 / 2, 0],
                               'Z': [0, 0, 1 / 2]}}
