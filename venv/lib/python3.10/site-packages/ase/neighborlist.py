# fmt: off

import itertools

import numpy as np
import scipy.sparse.csgraph as csgraph
from scipy import sparse as sp
from scipy.spatial import cKDTree

from ase.cell import Cell
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import (
    complete_cell,
    find_mic,
    minkowski_reduce,
    wrap_positions,
)
from ase.utils import deprecated


def natural_cutoffs(atoms, mult=1, **kwargs):
    """Generate a radial cutoff for every atom based on covalent radii.

    The covalent radii are a reasonable cutoff estimation for bonds in
    many applications such as neighborlists, so function generates an
    atoms length list of radii based on this idea.

    * atoms: An atoms object
    * mult: A multiplier for all cutoffs, useful for coarse grained adjustment
    * kwargs: Symbol of the atom and its corresponding cutoff,
      used to override the covalent radii
    """
    return [kwargs.get(atom.symbol, covalent_radii[atom.number] * mult)
            for atom in atoms]


def build_neighbor_list(atoms, cutoffs=None, **kwargs):
    """Automatically build and update a NeighborList.

    Parameters:

    atoms : :class:`~ase.Atoms` object
        Atoms to build Neighborlist for.
    cutoffs: list of floats
        Radii for each atom. If not given it will be produced by calling
        :func:`ase.neighborlist.natural_cutoffs`
    kwargs: arbitrary number of options
        Will be passed to the constructor of
        :class:`~ase.neighborlist.NeighborList`

    Returns:

    return: :class:`~ase.neighborlist.NeighborList`
        A :class:`~ase.neighborlist.NeighborList` instance (updated).
    """
    if cutoffs is None:
        cutoffs = natural_cutoffs(atoms)

    nl = NeighborList(cutoffs, **kwargs)
    nl.update(atoms)

    return nl


def get_distance_matrix(graph, limit=3):
    """Get Distance Matrix from a Graph.

    Parameters:

    graph: array, matrix or sparse matrix, 2 dimensions (N, N)
        Graph representation of the connectivity.
        See `scipy doc <https://docs.scipy.org/doc/scipy/reference/generated\
/scipy.sparse.csgraph.dijkstra.html#scipy.sparse.csgraph.dijkstra>`_
        for reference.
    limit: integer
        Maximum number of steps to analyze. For most molecular information,
        three should be enough.

    Returns:

    return: scipy.sparse.csr_matrix, shape (N, N)
        A scipy.sparse.csr_matrix. All elements that are not connected within
        *limit* steps are set to zero.

    This is a potential memory bottleneck, as csgraph.dijkstra produces a
    dense output matrix. Here we replace all np.inf values with 0 and
    transform back to csr_matrix.
    Why not dok_matrix like the connectivity-matrix? Because row-picking
    is most likely and this is super fast with csr.
    """
    mat = csgraph.dijkstra(graph, directed=False, limit=limit)
    mat[mat == np.inf] = 0
    return sp.csr_matrix(mat, dtype=np.int8)


def get_distance_indices(distanceMatrix, distance):
    """Get indices for each node that are distance or less away.

    Parameters:

    distanceMatrix: any one of scipy.sparse matrices (NxN)
        Matrix containing distance information of atoms. Get it e.g. with
        :func:`~ase.neighborlist.get_distance_matrix`
    distance: integer
        Number of steps to allow.

    Returns:

    return: list of length N
        List of length N. return[i] has all indices connected to item i.

    The distance matrix only contains shortest paths, so when looking for
    distances longer than one, we need to add the lower values for cases
    where atoms are connected via a shorter path too.
    """
    shape = distanceMatrix.get_shape()
    indices = []
    # iterate over rows
    for i in range(shape[0]):
        row = distanceMatrix.getrow(i)[0]
        # find all non-zero
        found = sp.find(row)
        # screen for smaller or equal distance
        equal = np.where(found[-1] <= distance)[0]
        # found[1] contains the indexes
        indices.append([found[1][x] for x in equal])
    return indices


def mic(dr, cell, pbc=True):
    """
    Apply minimum image convention to an array of distance vectors.

    Parameters:

    dr : array_like
        Array of distance vectors.
    cell : array_like
        Simulation cell.
    pbc : array_like, optional
        Periodic boundary conditions in x-, y- and z-direction. Default is to
        assume periodic boundaries in all directions.

    Returns:

    dr : array
        Array of distance vectors, wrapped according to the minimum image
        convention.
    """
    dr, _ = find_mic(dr, cell, pbc)
    return dr


def primitive_neighbor_list(quantities, pbc, cell, positions, cutoff,
                            numbers=None, self_interaction=False,
                            use_scaled_positions=False, max_nbins=1e6):
    """Compute a neighbor list for an atomic configuration.

    Atoms outside periodic boundaries are mapped into the box. Atoms
    outside nonperiodic boundaries are included in the neighbor list
    but complexity of neighbor list search for those can become n^2.

    The neighbor list is sorted by first atom index 'i', but not by second
    atom index 'j'.

    Parameters:

    quantities: str
        Quantities to compute by the neighbor list algorithm. Each character
        in this string defines a quantity. They are returned in a tuple of
        the same order. Possible quantities are

            * 'i' : first atom index
            * 'j' : second atom index
            * 'd' : absolute distance
            * 'D' : distance vector
            * 'S' : shift vector (number of cell boundaries crossed by the bond
              between atom i and j). With the shift vector S, the
              distances D between atoms can be computed from:
              D = positions[j]-positions[i]+S.dot(cell)
    pbc: array_like
        3-tuple indicating giving periodic boundaries in the three Cartesian
        directions.
    cell: 3x3 matrix
        Unit cell vectors.
    positions: list of xyz-positions
        Atomic positions.  Anything that can be converted to an ndarray of
        shape (n, 3) will do: [(x1,y1,z1), (x2,y2,z2), ...]. If
        use_scaled_positions is set to true, this must be scaled positions.
    cutoff: float or dict
        Cutoff for neighbor search. It can be:

            * A single float: This is a global cutoff for all elements.
            * A dictionary: This specifies cutoff values for element
              pairs. Specification accepts element numbers of symbols.
              Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
            * A list/array with a per atom value: This specifies the radius of
              an atomic sphere for each atoms. If spheres overlap, atoms are
              within each others neighborhood. See
              :func:`~ase.neighborlist.natural_cutoffs`
              for an example on how to get such a list.
    self_interaction: bool
        Return the atom itself as its own neighbor if set to true.
        Default: False
    use_scaled_positions: bool
        If set to true, positions are expected to be scaled positions.
    max_nbins: int
        Maximum number of bins used in neighbor search. This is used to limit
        the maximum amount of memory required by the neighbor list.

    Returns:

    i, j, ... : array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a)-1, but the order of (i,j)
        pairs is not guaranteed.

    """

    # Naming conventions: Suffixes indicate the dimension of an array. The
    # following convention is used here:
    #     c: Cartesian index, can have values 0, 1, 2
    #     i: Global atom index, can have values 0..len(a)-1
    #     xyz: Bin index, three values identifying x-, y- and z-component of a
    #          spatial bin that is used to make neighbor search O(n)
    #     b: Linearized version of the 'xyz' bin index
    #     a: Bin-local atom index, i.e. index identifying an atom *within* a
    #        bin
    #     p: Pair index, can have value 0 or 1
    #     n: (Linear) neighbor index

    # Return empty neighbor list if no atoms are passed here
    if len(positions) == 0:
        empty_types = dict(i=(int, (0, )),
                           j=(int, (0, )),
                           D=(float, (0, 3)),
                           d=(float, (0, )),
                           S=(int, (0, 3)))
        retvals = []
        for i in quantities:
            dtype, shape = empty_types[i]
            retvals += [np.array([], dtype=dtype).reshape(shape)]
        if len(retvals) == 1:
            return retvals[0]
        else:
            return tuple(retvals)

    # Compute reciprocal lattice vectors.
    b1_c, b2_c, b3_c = np.linalg.pinv(cell).T

    # Compute distances of cell faces.
    l1 = np.linalg.norm(b1_c)
    l2 = np.linalg.norm(b2_c)
    l3 = np.linalg.norm(b3_c)
    face_dist_c = np.array([1 / l1 if l1 > 0 else 1,
                            1 / l2 if l2 > 0 else 1,
                            1 / l3 if l3 > 0 else 1])

    if isinstance(cutoff, dict):
        max_cutoff = max(cutoff.values())
    else:
        if np.isscalar(cutoff):
            max_cutoff = cutoff
        else:
            cutoff = np.asarray(cutoff)
            max_cutoff = 2 * np.max(cutoff)

    # We use a minimum bin size of 3 A
    bin_size = max(max_cutoff, 3)
    # Compute number of bins such that a sphere of radius cutoff fits into
    # eight neighboring bins.
    nbins_c = np.maximum((face_dist_c / bin_size).astype(int), [1, 1, 1])
    nbins = np.prod(nbins_c)
    # Make sure we limit the amount of memory used by the explicit bins.
    while nbins > max_nbins:
        nbins_c = np.maximum(nbins_c // 2, [1, 1, 1])
        nbins = np.prod(nbins_c)

    # Compute over how many bins we need to loop in the neighbor list search.
    neigh_search_x, neigh_search_y, neigh_search_z = \
        np.ceil(bin_size * nbins_c / face_dist_c).astype(int)

    # If we only have a single bin and the system is not periodic, then we
    # do not need to search neighboring bins
    neigh_search_x = 0 if nbins_c[0] == 1 and not pbc[0] else neigh_search_x
    neigh_search_y = 0 if nbins_c[1] == 1 and not pbc[1] else neigh_search_y
    neigh_search_z = 0 if nbins_c[2] == 1 and not pbc[2] else neigh_search_z

    # Sort atoms into bins.
    if use_scaled_positions:
        scaled_positions_ic = positions
        positions = np.dot(scaled_positions_ic, cell)
    else:
        scaled_positions_ic = np.linalg.solve(complete_cell(cell).T,
                                              positions.T).T
    bin_index_ic = np.floor(scaled_positions_ic * nbins_c).astype(int)
    cell_shift_ic = np.zeros_like(bin_index_ic)

    for c in range(3):
        if pbc[c]:
            # (Note: np.divmod does not exist in older numpies)
            cell_shift_ic[:, c], bin_index_ic[:, c] = \
                divmod(bin_index_ic[:, c], nbins_c[c])
        else:
            bin_index_ic[:, c] = np.clip(bin_index_ic[:, c], 0, nbins_c[c] - 1)

    # Convert Cartesian bin index to unique scalar bin index.
    bin_index_i = (bin_index_ic[:, 0] +
                   nbins_c[0] * (bin_index_ic[:, 1] +
                                 nbins_c[1] * bin_index_ic[:, 2]))

    # atom_i contains atom index in new sort order.
    atom_i = np.argsort(bin_index_i)
    bin_index_i = bin_index_i[atom_i]

    # Find max number of atoms per bin
    max_natoms_per_bin = np.bincount(bin_index_i).max()

    # Sort atoms into bins: atoms_in_bin_ba contains for each bin (identified
    # by its scalar bin index) a list of atoms inside that bin. This list is
    # homogeneous, i.e. has the same size *max_natoms_per_bin* for all bins.
    # The list is padded with -1 values.
    atoms_in_bin_ba = -np.ones([nbins, max_natoms_per_bin], dtype=int)
    for i in range(max_natoms_per_bin):
        # Create a mask array that identifies the first atom of each bin.
        mask = np.append([True], bin_index_i[:-1] != bin_index_i[1:])
        # Assign all first atoms.
        atoms_in_bin_ba[bin_index_i[mask], i] = atom_i[mask]

        # Remove atoms that we just sorted into atoms_in_bin_ba. The next
        # "first" atom will be the second and so on.
        mask = np.logical_not(mask)
        atom_i = atom_i[mask]
        bin_index_i = bin_index_i[mask]

    # Make sure that all atoms have been sorted into bins.
    assert len(atom_i) == 0
    assert len(bin_index_i) == 0

    # Now we construct neighbor pairs by pairing up all atoms within a bin or
    # between bin and neighboring bin. atom_pairs_pn is a helper buffer that
    # contains all potential pairs of atoms between two bins, i.e. it is a list
    # of length max_natoms_per_bin**2.
    atom_pairs_pn = np.indices((max_natoms_per_bin, max_natoms_per_bin),
                               dtype=int)
    atom_pairs_pn = atom_pairs_pn.reshape(2, -1)

    # Initialized empty neighbor list buffers.
    first_at_neightuple_nn = []
    secnd_at_neightuple_nn = []
    cell_shift_vector_x_n = []
    cell_shift_vector_y_n = []
    cell_shift_vector_z_n = []

    # This is the main neighbor list search. We loop over neighboring bins and
    # then construct all possible pairs of atoms between two bins, assuming
    # that each bin contains exactly max_natoms_per_bin atoms. We then throw
    # out pairs involving pad atoms with atom index -1 below.
    binz_xyz, biny_xyz, binx_xyz = np.meshgrid(np.arange(nbins_c[2]),
                                               np.arange(nbins_c[1]),
                                               np.arange(nbins_c[0]),
                                               indexing='ij')
    # The memory layout of binx_xyz, biny_xyz, binz_xyz is such that computing
    # the respective bin index leads to a linearly increasing consecutive list.
    # The following assert statement succeeds:
    #     b_b = (binx_xyz + nbins_c[0] * (biny_xyz + nbins_c[1] *
    #                                     binz_xyz)).ravel()
    #     assert (b_b == np.arange(np.prod(nbins_c))).all()

    # First atoms in pair.
    _first_at_neightuple_n = atoms_in_bin_ba[:, atom_pairs_pn[0]]
    for dz in range(-neigh_search_z, neigh_search_z + 1):
        for dy in range(-neigh_search_y, neigh_search_y + 1):
            for dx in range(-neigh_search_x, neigh_search_x + 1):
                # Bin index of neighboring bin and shift vector.
                shiftx_xyz, neighbinx_xyz = divmod(binx_xyz + dx, nbins_c[0])
                shifty_xyz, neighbiny_xyz = divmod(biny_xyz + dy, nbins_c[1])
                shiftz_xyz, neighbinz_xyz = divmod(binz_xyz + dz, nbins_c[2])
                neighbin_b = (neighbinx_xyz + nbins_c[0] *
                              (neighbiny_xyz + nbins_c[1] * neighbinz_xyz)
                              ).ravel()

                # Second atom in pair.
                _secnd_at_neightuple_n = \
                    atoms_in_bin_ba[neighbin_b][:, atom_pairs_pn[1]]

                # Shift vectors.
                _cell_shift_vector_x_n = \
                    np.resize(shiftx_xyz.reshape(-1, 1),
                              (max_natoms_per_bin**2, shiftx_xyz.size)).T
                _cell_shift_vector_y_n = \
                    np.resize(shifty_xyz.reshape(-1, 1),
                              (max_natoms_per_bin**2, shifty_xyz.size)).T
                _cell_shift_vector_z_n = \
                    np.resize(shiftz_xyz.reshape(-1, 1),
                              (max_natoms_per_bin**2, shiftz_xyz.size)).T

                # We have created too many pairs because we assumed each bin
                # has exactly max_natoms_per_bin atoms. Remove all surperfluous
                # pairs. Those are pairs that involve an atom with index -1.
                mask = np.logical_and(_first_at_neightuple_n != -1,
                                      _secnd_at_neightuple_n != -1)
                if mask.sum() > 0:
                    first_at_neightuple_nn += [_first_at_neightuple_n[mask]]
                    secnd_at_neightuple_nn += [_secnd_at_neightuple_n[mask]]
                    cell_shift_vector_x_n += [_cell_shift_vector_x_n[mask]]
                    cell_shift_vector_y_n += [_cell_shift_vector_y_n[mask]]
                    cell_shift_vector_z_n += [_cell_shift_vector_z_n[mask]]

    # Flatten overall neighbor list.
    first_at_neightuple_n = np.concatenate(first_at_neightuple_nn)
    secnd_at_neightuple_n = np.concatenate(secnd_at_neightuple_nn)
    cell_shift_vector_n = np.transpose([np.concatenate(cell_shift_vector_x_n),
                                        np.concatenate(cell_shift_vector_y_n),
                                        np.concatenate(cell_shift_vector_z_n)])

    # Add global cell shift to shift vectors
    cell_shift_vector_n += cell_shift_ic[first_at_neightuple_n] - \
        cell_shift_ic[secnd_at_neightuple_n]

    # Remove all self-pairs that do not cross the cell boundary.
    if not self_interaction:
        m = np.logical_not(np.logical_and(
            first_at_neightuple_n == secnd_at_neightuple_n,
            (cell_shift_vector_n == 0).all(axis=1)))
        first_at_neightuple_n = first_at_neightuple_n[m]
        secnd_at_neightuple_n = secnd_at_neightuple_n[m]
        cell_shift_vector_n = cell_shift_vector_n[m]

    # For nonperiodic directions, remove any bonds that cross the domain
    # boundary.
    for c in range(3):
        if not pbc[c]:
            m = cell_shift_vector_n[:, c] == 0
            first_at_neightuple_n = first_at_neightuple_n[m]
            secnd_at_neightuple_n = secnd_at_neightuple_n[m]
            cell_shift_vector_n = cell_shift_vector_n[m]

    # Sort neighbor list.
    i = np.argsort(first_at_neightuple_n)
    first_at_neightuple_n = first_at_neightuple_n[i]
    secnd_at_neightuple_n = secnd_at_neightuple_n[i]
    cell_shift_vector_n = cell_shift_vector_n[i]

    # Compute distance vectors.
    distance_vector_nc = positions[secnd_at_neightuple_n] - \
        positions[first_at_neightuple_n] + \
        cell_shift_vector_n.dot(cell)
    abs_distance_vector_n = \
        np.sqrt(np.sum(distance_vector_nc * distance_vector_nc, axis=1))

    # We have still created too many pairs. Only keep those with distance
    # smaller than max_cutoff.
    mask = abs_distance_vector_n < max_cutoff
    first_at_neightuple_n = first_at_neightuple_n[mask]
    secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
    cell_shift_vector_n = cell_shift_vector_n[mask]
    distance_vector_nc = distance_vector_nc[mask]
    abs_distance_vector_n = abs_distance_vector_n[mask]

    if isinstance(cutoff, dict) and numbers is not None:
        # If cutoff is a dictionary, then the cutoff radii are specified per
        # element pair. We now have a list up to maximum cutoff.
        per_pair_cutoff_n = np.zeros_like(abs_distance_vector_n)
        for (atomic_number1, atomic_number2), c in cutoff.items():
            try:
                atomic_number1 = atomic_numbers[atomic_number1]
            except KeyError:
                pass
            try:
                atomic_number2 = atomic_numbers[atomic_number2]
            except KeyError:
                pass
            if atomic_number1 == atomic_number2:
                mask = np.logical_and(
                    numbers[first_at_neightuple_n] == atomic_number1,
                    numbers[secnd_at_neightuple_n] == atomic_number2)
            else:
                mask = np.logical_or(
                    np.logical_and(
                        numbers[first_at_neightuple_n] == atomic_number1,
                        numbers[secnd_at_neightuple_n] == atomic_number2),
                    np.logical_and(
                        numbers[first_at_neightuple_n] == atomic_number2,
                        numbers[secnd_at_neightuple_n] == atomic_number1))
            per_pair_cutoff_n[mask] = c
        mask = abs_distance_vector_n < per_pair_cutoff_n
        first_at_neightuple_n = first_at_neightuple_n[mask]
        secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
        cell_shift_vector_n = cell_shift_vector_n[mask]
        distance_vector_nc = distance_vector_nc[mask]
        abs_distance_vector_n = abs_distance_vector_n[mask]
    elif not np.isscalar(cutoff):
        # If cutoff is neither a dictionary nor a scalar, then we assume it is
        # a list or numpy array that contains atomic radii. Atoms are neighbors
        # if their radii overlap.
        mask = abs_distance_vector_n < \
            cutoff[first_at_neightuple_n] + cutoff[secnd_at_neightuple_n]
        first_at_neightuple_n = first_at_neightuple_n[mask]
        secnd_at_neightuple_n = secnd_at_neightuple_n[mask]
        cell_shift_vector_n = cell_shift_vector_n[mask]
        distance_vector_nc = distance_vector_nc[mask]
        abs_distance_vector_n = abs_distance_vector_n[mask]

    # Assemble return tuple.
    retvals = []
    for q in quantities:
        if q == 'i':
            retvals += [first_at_neightuple_n]
        elif q == 'j':
            retvals += [secnd_at_neightuple_n]
        elif q == 'D':
            retvals += [distance_vector_nc]
        elif q == 'd':
            retvals += [abs_distance_vector_n]
        elif q == 'S':
            retvals += [cell_shift_vector_n]
        else:
            raise ValueError('Unsupported quantity specified.')
    if len(retvals) == 1:
        return retvals[0]
    else:
        return tuple(retvals)


def neighbor_list(quantities, a, cutoff, self_interaction=False,
                  max_nbins=1e6):
    """Compute a neighbor list for an atomic configuration.

    Atoms outside periodic boundaries are mapped into the box. Atoms
    outside nonperiodic boundaries are included in the neighbor list
    but complexity of neighbor list search for those can become n^2.

    The neighbor list is sorted by first atom index 'i', but not by second
    atom index 'j'.

    Parameters
    ----------
    quantities: str
        Quantities to compute by the neighbor list algorithm. Each character
        in this string defines a quantity. They are returned in a tuple of
        the same order. Possible quantities are:

           * 'i' : first atom index
           * 'j' : second atom index
           * 'd' : absolute distance
           * 'D' : distance vector
           * 'S' : shift vector (number of cell boundaries crossed by the bond
             between atom i and j). With the shift vector S, the
             distances D between atoms can be computed from:
             D = a.positions[j]-a.positions[i]+S.dot(a.cell)
    a: :class:`ase.Atoms`
        Atomic configuration.
    cutoff: float or dict
        Cutoff for neighbor search. It can be:

            * A single float: This is a global cutoff for all elements.
            * A dictionary: This specifies cutoff values for element
              pairs. Specification accepts element numbers of symbols.
              Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
            * A list/array with a per atom value: This specifies the radius of
              an atomic sphere for each atoms. If spheres overlap, atoms are
              within each others neighborhood. See
              :func:`~ase.neighborlist.natural_cutoffs`
              for an example on how to get such a list.

    self_interaction: bool
        Return the atom itself as its own neighbor if set to true.
        Default: False
    max_nbins: int
        Maximum number of bins used in neighbor search. This is used to limit
        the maximum amount of memory required by the neighbor list.

    Returns
    -------
    i, j, ...: array
        Tuple with arrays for each quantity specified above. Indices in `i`
        are returned in ascending order 0..len(a), but the order of (i,j)
        pairs is not guaranteed.

    Examples
    --------

    >>> import numpy as np
    >>> from ase.build import bulk, molecule

    1. Coordination counting

    >>> atoms = molecule('isobutane')
    >>> i = neighbor_list('i', atoms, 1.85)
    >>> coord = np.bincount(i, minlength=len(atoms))

    2. Coordination counting with different cutoffs for each pair of species

    >>> cutoff = {('H', 'H'): 1.1, ('C', 'H'): 1.3, ('C', 'C'): 1.85}
    >>> i = neighbor_list('i', atoms, cutoff)
    >>> coord = np.bincount(i, minlength=len(atoms))

    3. Pair distribution function

    >>> atoms = bulk('Cu', cubic=True) * 3
    >>> atoms.rattle(0.5, rng=np.random.default_rng(42))
    >>> cutoff = 5.0
    >>> d = neighbor_list('d', atoms, cutoff)
    >>> hist, bin_edges = np.histogram(d, bins=100, range=(0.0, cutoff))
    >>> hist = hist / len(atoms)  # per atom
    >>> rho_mean = len(atoms) / atoms.cell.volume
    >>> dv = 4.0 * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3) / 3.0
    >>> rho = hist / dv
    >>> pdf = rho / rho_mean

    4. Forces of a pair potential

    >>> natoms = len(atoms)
    >>> i, j, d, D = neighbor_list('ijdD', atoms, 5.0)
    >>> # Lennard-Jones potential
    >>> eps = 1.0
    >>> sgm = 1.0
    >>> epairs = 4.0 * eps * ((sgm / d) ** 12 - (sgm / d) ** 6)
    >>> energy = 0.5 * epairs.sum()  # correct double-counting
    >>> dd = -4.0 * eps * (12 * (sgm / d) ** 13 - 6 * (sgm / d) ** 7) / sgm
    >>> dd = (dd * (D.T / d)).T
    >>> fx = -1.0 * np.bincount(i, weights=dd[:, 0], minlength=natoms)
    >>> fy = -1.0 * np.bincount(i, weights=dd[:, 1], minlength=natoms)
    >>> fz = -1.0 * np.bincount(i, weights=dd[:, 2], minlength=natoms)

    5. Force-constant matrix of a pair potential

    >>> i, j, d, D = neighbor_list('ijdD', atoms, 5.0)
    >>> epairs = 1.0 / d  # Coulomb potential
    >>> forces = (D.T / d**3).T  # (npairs, 3)
    >>> # second derivative
    >>> d2 = 3.0 * D[:, :, None] * D[:, None, :] / d[:, None, None] ** 5
    >>> for k in range(3):
    ...     d2[:, k, k] -= 1.0 / d**3
    >>> # force-constant matrix
    >>> fc = np.zeros((natoms, 3, natoms, 3))
    >>> for ia in range(natoms):
    ...     for ja in range(natoms):
    ...         fc[ia, :, ja, :] -= d2[(i == ia) & (j == ja), :, :].sum(axis=0)
    >>> for ia in range(natoms):
    ...     fc[ia, :, ia, :] -= fc[ia].sum(axis=1)

    """
    return primitive_neighbor_list(quantities, a.pbc,
                                   a.get_cell(complete=True),
                                   a.positions, cutoff, numbers=a.numbers,
                                   self_interaction=self_interaction,
                                   max_nbins=max_nbins)


def first_neighbors(natoms, first_atom):
    """
    Compute an index array pointing to the ranges within the neighbor list that
    contain the neighbors for a certain atom.

    Parameters:

    natoms : int
        Total number of atom.
    first_atom : array_like
        Array containing the first atom 'i' of the neighbor tuple returned
        by the neighbor list.

    Returns:

    seed : array
        Array containing pointers to the start and end location of the
        neighbors of a certain atom. Neighbors of atom k have indices from s[k]
        to s[k+1]-1.
    """
    if len(first_atom) == 0:
        return np.zeros(natoms + 1, dtype=int)
    # Create a seed array (which is returned by this function) populated with
    # -1.
    seed = -np.ones(natoms + 1, dtype=int)

    first_atom = np.asarray(first_atom)

    # Mask array contains all position where the number in the (sorted) array
    # with first atoms (in the neighbor pair) changes.
    mask = first_atom[:-1] != first_atom[1:]

    # Seed array needs to start at 0
    seed[first_atom[0]] = 0
    # Seed array needs to stop at the length of the neighbor list
    seed[-1] = len(first_atom)
    # Populate all intermediate seed with the index of where the mask array is
    # true, i.e. the index where the first_atom array changes.
    seed[first_atom[1:][mask]] = (np.arange(len(mask)) + 1)[mask]

    # Now fill all remaining -1 value with the value in the seed array right
    # behind them. (There are no neighbor so seed[i] and seed[i+1] must point)
    # to the same index.
    mask = seed == -1
    while mask.any():
        seed[mask] = seed[np.arange(natoms + 1)[mask] + 1]
        mask = seed == -1
    return seed


def get_connectivity_matrix(nl, sparse=True):
    """Return connectivity matrix for a given NeighborList (dtype=numpy.int8).

    A matrix of shape (nAtoms, nAtoms) will be returned.
    Connected atoms i and j will have matrix[i,j] == 1, unconnected
    matrix[i,j] == 0. If bothways=True the matrix will be symmetric,
    otherwise not!

    If *sparse* is True, a scipy csr matrix is returned.
    If *sparse* is False, a numpy matrix is returned.

    Note that the old and new neighborlists might give different results
    for periodic systems if bothways=False.

    Example:

    Determine which molecule in a system atom 1 belongs to.

    >>> from scipy import sparse

    >>> from ase.build import molecule
    >>> from ase.neighborlist import get_connectivity_matrix
    >>> from ase.neighborlist import natural_cutoffs
    >>> from ase.neighborlist import NeighborList

    >>> mol = molecule('CH3CH2OH')
    >>> cutoffs = natural_cutoffs(mol)
    >>> neighbor_list = NeighborList(
    ...     cutoffs, self_interaction=False, bothways=True)
    >>> neighbor_list.update(mol)
    True

    >>> matrix = neighbor_list.get_connectivity_matrix()
    >>> # or: matrix = get_connectivity_matrix(neighbor_list.nl)
    >>> n_components, component_list = sparse.csgraph.connected_components(
    ...    matrix)
    >>> idx = 1
    >>> molIdx = component_list[idx]
    >>> print("There are {} molecules in the system".format(n_components))
    There are 1 molecules in the system
    >>> print("Atom {} is part of molecule {}".format(idx, molIdx))
    Atom 1 is part of molecule 0
    >>> molIdxs = [i for i in range(len(component_list))
    ...            if component_list[i] == molIdx]
    >>> print("Atoms are part of molecule {}: {}".format(molIdx, molIdxs))
    Atoms are part of molecule 0: [0, 1, 2, 3, 4, 5, 6, 7, 8]
    """

    nAtoms = len(nl.cutoffs)

    if nl.nupdates <= 0:
        raise RuntimeError(
            'Must call update(atoms) on your neighborlist first!')

    if sparse:
        matrix = sp.dok_matrix((nAtoms, nAtoms), dtype=np.int8)
    else:
        matrix = np.zeros((nAtoms, nAtoms), dtype=np.int8)

    for i in range(nAtoms):
        for idx in nl.get_neighbors(i)[0]:
            matrix[i, idx] = 1

    return matrix


class NewPrimitiveNeighborList:
    """Neighbor list object. Wrapper around neighbor_list and first_neighbors.

    cutoffs: list of float
        List of cutoff radii - one for each atom. If the spheres (defined by
        their cutoff radii) of two atoms overlap, they will be counted as
        neighbors.
    skin: float
        If no atom has moved more than the skin-distance since the
        last call to the
        :meth:`~ase.neighborlist.NewPrimitiveNeighborList.update()`
        method, then the neighbor list can be reused. This will save
        some expensive rebuilds of the list, but extra neighbors outside
        the cutoff will be returned.
    sorted: bool
        Sort neighbor list.
    self_interaction: bool
        Should an atom return itself as a neighbor?
    bothways: bool
        Return all neighbors.  Default is to return only "half" of
        the neighbors.

    Example:

    >>> from ase.build import bulk
    >>> from ase.neighborlist import NewPrimitiveNeighborList

    >>> nl = NewPrimitiveNeighborList([2.3, 1.7])
    >>> atoms = bulk('Cu', 'fcc', a=3.6)
    >>> nl.update(atoms.pbc, atoms.get_cell(), atoms.positions)
    True
    >>> indices, offsets = nl.get_neighbors(0)
    """

    def __init__(self, cutoffs, skin=0.3, sorted=False, self_interaction=True,
                 bothways=False, use_scaled_positions=False):
        self.cutoffs = np.asarray(cutoffs) + skin
        self.skin = skin
        self.sorted = sorted
        self.self_interaction = self_interaction
        self.bothways = bothways
        self.nupdates = 0
        self.use_scaled_positions = use_scaled_positions

    def update(self, pbc, cell, positions, numbers=None):
        """Make sure the list is up to date."""

        if self.nupdates == 0:
            self.build(pbc, cell, positions, numbers=numbers)
            return True

        if ((self.pbc != pbc).any() or (self.cell != cell).any() or
                ((self.positions - positions)**2).sum(1).max() > self.skin**2):
            self.build(pbc, cell, positions, numbers=numbers)
            return True

        return False

    def build(self, pbc, cell, positions, numbers=None):
        """Build the list.
        """
        self.pbc = np.array(pbc, copy=True)
        self.cell = np.array(cell, copy=True)
        self.positions = np.array(positions, copy=True)

        pair_first, pair_second, offset_vec = \
            primitive_neighbor_list(
                'ijS', pbc, cell, positions, self.cutoffs, numbers=numbers,
                self_interaction=self.self_interaction,
                use_scaled_positions=self.use_scaled_positions)

        if len(positions) > 0 and not self.bothways:
            offset_x, offset_y, offset_z = offset_vec.T

            mask = offset_z > 0
            mask &= offset_y == 0
            mask |= offset_y > 0
            mask &= offset_x == 0
            mask |= offset_x > 0
            mask |= (pair_first <= pair_second) & (offset_vec == 0).all(axis=1)

            pair_first = pair_first[mask]
            pair_second = pair_second[mask]
            offset_vec = offset_vec[mask]

        if len(positions) > 0 and self.sorted:
            mask = np.argsort(pair_first * len(pair_first) +
                              pair_second)
            pair_first = pair_first[mask]
            pair_second = pair_second[mask]
            offset_vec = offset_vec[mask]

        self.pair_first = pair_first
        self.pair_second = pair_second
        self.offset_vec = offset_vec

        # Compute the index array point to the first neighbor
        self.first_neigh = first_neighbors(len(positions), pair_first)

        self.nupdates += 1

    def get_neighbors(self, a):
        """Return neighbors of atom number a.

        A list of indices and offsets to neighboring atoms is
        returned.  The positions of the neighbor atoms can be
        calculated like this:

        >>> from ase.build import bulk
        >>> from ase.neighborlist import NewPrimitiveNeighborList

        >>> nl = NewPrimitiveNeighborList([2.3, 1.7])
        >>> atoms = bulk('Cu', 'fcc', a=3.6)
        >>> nl.update(atoms.pbc, atoms.get_cell(), atoms.positions)
        True
        >>> indices, offsets = nl.get_neighbors(0)
        >>> for i, offset in zip(indices, offsets):
        ...     print(atoms.positions[i] + offset @ atoms.get_cell())
        ... # doctest: +SKIP

        Notice that if get_neighbors(a) gives atom b as a neighbor,
        then get_neighbors(b) will not return a as a neighbor - unless
        bothways=True was used."""

        return (self.pair_second[self.first_neigh[a]:self.first_neigh[a + 1]],
                self.offset_vec[self.first_neigh[a]:self.first_neigh[a + 1]])


class PrimitiveNeighborList:
    """Neighbor list that works without Atoms objects.

    This is less fancy, but can be used to avoid conversions between
    scaled and non-scaled coordinates which may affect cell offsets
    through rounding errors.

    Attributes
    ----------
    nupdates : int
        Number of updated times.
    """

    def __init__(self, cutoffs, skin=0.3, sorted=False, self_interaction=True,
                 bothways=False, use_scaled_positions=False):
        self.cutoffs = np.asarray(cutoffs) + skin
        self.skin = skin
        self.sorted = sorted
        self.self_interaction = self_interaction
        self.bothways = bothways
        self.nupdates = 0
        self.use_scaled_positions = use_scaled_positions

    def update(self, pbc, cell, coordinates):
        """Make sure the list is up to date.

        Returns
        -------
        bool
            True if the neighbor list is updated.
        """

        if self.nupdates == 0:
            self.build(pbc, cell, coordinates)
            return True

        if ((self.pbc != pbc).any() or (self.cell != cell).any() or (
                (self.coordinates
                 - coordinates)**2).sum(1).max() > self.skin**2):
            self.build(pbc, cell, coordinates)
            return True

        return False

    def build(self, pbc, cell, coordinates):
        """Build the list.

        Coordinates are taken to be scaled or not according
        to self.use_scaled_positions.
        """
        self.pbc = pbc = np.array(pbc, copy=True)
        self.cell = cell = Cell(cell)
        self.coordinates = coordinates = np.array(coordinates, copy=True)

        if len(self.cutoffs) != len(coordinates):
            raise ValueError('Wrong number of cutoff radii: {} != {}'
                             .format(len(self.cutoffs), len(coordinates)))

        rcmax = self.cutoffs.max() if len(self.cutoffs) > 0 else 0.0

        if self.use_scaled_positions:
            positions0 = cell.cartesian_positions(coordinates)
        else:
            positions0 = coordinates

        rcell, op = minkowski_reduce(cell, pbc)
        positions = wrap_positions(positions0, rcell, pbc=pbc, eps=0)

        natoms = len(positions)
        self.nupdates += 1
        if natoms == 0:
            self.neighbors = []
            self.displacements = []
            return

        tree = cKDTree(positions, copy_data=True)
        offsets = cell.scaled_positions(positions - positions0)
        offsets = offsets.round().astype(int)

        N = _calc_expansion(rcell, pbc, rcmax)

        neighbor_indices_a = [[] for _ in range(natoms)]
        displacements_a = [[] for _ in range(natoms)]

        for n1, n2, n3 in itertools.product(range(N[0] + 1),
                                            range(-N[1], N[1] + 1),
                                            range(-N[2], N[2] + 1)):
            if n1 == 0 and (n2 < 0 or n2 == 0 and n3 < 0):
                continue

            displacement = (n1, n2, n3) @ rcell
            shift0 = (n1, n2, n3) @ op
            indices_all = tree.query_ball_point(
                positions - displacement,
                r=self.cutoffs + rcmax,
            )

            for a in range(natoms):
                indices = indices_all[a]

                if not indices:
                    continue

                indices = np.array(indices)
                delta = positions[indices] + displacement - positions[a]
                distances = np.sqrt(np.add.reduce(delta**2, axis=1))
                cutoffs = self.cutoffs[indices] + self.cutoffs[a]
                i = indices[distances < cutoffs]
                if n1 == 0 and n2 == 0 and n3 == 0:
                    if self.self_interaction:
                        i = i[i >= a]
                    else:
                        i = i[i > a]

                neighbor_indices_a[a].append(i)

                disp = shift0 + offsets[i] - offsets[a]
                displacements_a[a].append(disp)

        self.neighbors = [np.concatenate(i) for i in neighbor_indices_a]
        self.displacements = [np.concatenate(d) for d in displacements_a]

        if self.bothways:
            neighbors2 = [[] for a in range(natoms)]
            displacements2 = [[] for a in range(natoms)]
            for a in range(natoms):
                for b, disp in zip(self.neighbors[a], self.displacements[a]):
                    # avoid double counting of self interaction
                    if a == b and (disp == 0).all():
                        continue
                    neighbors2[b].append(a)
                    displacements2[b].append(-disp)
            for a in range(natoms):
                nbs = np.concatenate((self.neighbors[a], neighbors2[a]))
                disp = np.array(list(self.displacements[a]) + displacements2[a])
                # Force correct type and shape for case of no neighbors:
                self.neighbors[a] = nbs.astype(int)
                self.displacements[a] = disp.astype(int).reshape((-1, 3))

        if self.sorted:
            _sort_neighbors(self.neighbors, self.displacements)

    def get_neighbors(self, a):
        """Return neighbors of atom number a.

        A list of indices and offsets to neighboring atoms is
        returned.  The positions of the neighbor atoms can be
        calculated like this:

        >>> from ase.build import bulk
        >>> from ase.neighborlist import NewPrimitiveNeighborList

        >>> nl = NewPrimitiveNeighborList([2.3, 1.7])
        >>> atoms = bulk('Cu', 'fcc', a=3.6)
        >>> nl.update(atoms.pbc, atoms.get_cell(), atoms.positions)
        True
        >>> indices, offsets = nl.get_neighbors(0)
        >>> for i, offset in zip(indices, offsets):
        ...     print(atoms.positions[i] + offset @ atoms.get_cell())
        ... # doctest: +SKIP

        Notice that if get_neighbors(a) gives atom b as a neighbor,
        then get_neighbors(b) will not return a as a neighbor - unless
        bothways=True was used."""

        return self.neighbors[a], self.displacements[a]


def _calc_expansion(rcell, pbc, rcmax):
    r"""Calculate expansion to contain a sphere of radius `2.0 * rcmax`.

    This function determines the minimum supercell (parallelepiped) that
    contains a sphere of radius `2.0 * rcmax`. For this, `a_1` is projected
    onto the unit vector perpendicular to `a_2 \times a_3` (i.e. the unit
    vector along the direction `b_1`) to know how many `a_1`'s the supercell
    takes to contain the sphere.
    """
    ircell = np.linalg.pinv(rcell)
    vs = np.sqrt(np.add.reduce(ircell**2, axis=0))
    ns = np.where(pbc, np.ceil(2.0 * rcmax * vs), 0.0)
    return ns.astype(int)


def _sort_neighbors(neighbors, offsets):
    """Sort neighbors first by indices and then offsets."""
    natoms = len(neighbors)
    for a in range(natoms):
        keys = (
            offsets[a][:, 2],
            offsets[a][:, 1],
            offsets[a][:, 0],
            neighbors[a]
        )
        mask = np.lexsort(keys)
        neighbors[a] = neighbors[a][mask]
        offsets[a] = offsets[a][mask]


class NeighborList:
    """Neighbor list object.

    cutoffs: list of float
        List of cutoff radii - one for each atom. If the spheres
        (defined by their cutoff radii) of two atoms overlap, they
        will be counted as neighbors. See
        :func:`~ase.neighborlist.natural_cutoffs` for an example on
        how to get such a list.

    skin: float
        If no atom has moved more than the skin-distance since the
        last call to the
        :meth:`~ase.neighborlist.NeighborList.update()` method, then
        the neighbor list can be reused.  This will save some
        expensive rebuilds of the list, but extra neighbors outside
        the cutoff will be returned.
    self_interaction: bool
        Should an atom return itself as a neighbor?
    bothways: bool
        Return all neighbors.  Default is to return only "half" of
        the neighbors.
    primitive: class
        Define which implementation to use. Older and quadratically-scaling
        :class:`~ase.neighborlist.PrimitiveNeighborList` or newer and
        linearly-scaling :class:`~ase.neighborlist.NewPrimitiveNeighborList`.

    Example:

    >>> from ase.build import molecule
    >>> from ase.neighborlist import NeighborList

    >>> atoms = molecule("CO")
    >>> nl = NeighborList([0.76, 0.66])
    >>> nl.update(atoms)
    True
    >>> indices, offsets = nl.get_neighbors(0)

    """

    def __init__(self, cutoffs, skin=0.3, sorted=False, self_interaction=True,
                 bothways=False, primitive=PrimitiveNeighborList):
        self.nl = primitive(cutoffs, skin, sorted,
                            self_interaction=self_interaction,
                            bothways=bothways)

    def update(self, atoms):
        """
        See :meth:`ase.neighborlist.PrimitiveNeighborList.update` or
        :meth:`ase.neighborlist.PrimitiveNeighborList.update`.
        """
        return self.nl.update(atoms.pbc, atoms.get_cell(complete=True),
                              atoms.positions)

    def get_neighbors(self, a):
        """
        See :meth:`ase.neighborlist.PrimitiveNeighborList.get_neighbors` or
        :meth:`ase.neighborlist.PrimitiveNeighborList.get_neighbors`.
        """
        if self.nl.nupdates <= 0:
            raise RuntimeError('Must call update(atoms) on your neighborlist '
                               'first!')

        return self.nl.get_neighbors(a)

    def get_connectivity_matrix(self, sparse=True):
        """
        See :func:`~ase.neighborlist.get_connectivity_matrix`.
        """
        return get_connectivity_matrix(self.nl, sparse)

    @property
    def nupdates(self):
        """Get number of updates."""
        return self.nl.nupdates

    @property
    @deprecated(
        'Use, e.g., `sum(_.size for _ in nl.neighbors)` '
        'for `bothways=False` and `self_interaction=False`.'
    )
    def nneighbors(self):
        """Get number of neighbors.

        .. deprecated:: 3.24.0
        """
        nneighbors = sum(indices.size for indices in self.nl.neighbors)
        if self.nl.self_interaction:
            nneighbors -= len(self.nl.neighbors)
        return nneighbors // 2 if self.nl.bothways else nneighbors

    @property
    @deprecated(
        'Use, e.g., `sum(_.any(1).sum() for _ in nl.displacements)` '
        'for `bothways=False` and `self_interaction=False`.'
    )
    def npbcneighbors(self):
        """Get number of pbc neighbors.

        .. deprecated:: 3.24.0
        """
        nneighbors = sum(
            offsets.any(axis=1).sum() for offsets in self.nl.displacements
        )  # sum up all neighbors that have non-zero supercell offsets
        return nneighbors // 2 if self.nl.bothways else nneighbors
