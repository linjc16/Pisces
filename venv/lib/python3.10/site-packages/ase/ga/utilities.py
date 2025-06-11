# fmt: off

"""Various utility methods used troughout the GA."""
import itertools
import os
import time

import numpy as np
from scipy.spatial.distance import cdist

from ase.data import covalent_radii
from ase.ga import get_neighbor_list
from ase.geometry.cell import cell_to_cellpar
from ase.geometry.rdf import get_rdf
from ase.io import read, write


def closest_distances_generator(atom_numbers, ratio_of_covalent_radii):
    """Generates the blmin dict used across the GA.
    The distances are based on the covalent radii of the atoms.
    """
    cr = covalent_radii
    ratio = ratio_of_covalent_radii

    blmin = {}
    for i in atom_numbers:
        blmin[(i, i)] = cr[i] * 2 * ratio
        for j in atom_numbers:
            if i == j:
                continue
            if (i, j) in blmin:
                continue
            blmin[(i, j)] = blmin[(j, i)] = ratio * (cr[i] + cr[j])
    return blmin


def get_mic_distance(p1, p2, cell, pbc):
    """This method calculates the shortest distance between p1 and p2
    through the cell boundaries defined by cell and pbc.
    This method works for reasonable unit cells, but not for extremely
    elongated ones.
    """
    ct = cell.T
    pos = np.array((p1, p2))
    scaled = np.linalg.solve(ct, pos.T).T
    for i in range(3):
        if pbc[i]:
            scaled[:, i] %= 1.0
            scaled[:, i] %= 1.0
    P = np.dot(scaled, cell)

    pbc_directions = [[-1, 1] * int(direction) + [0] for direction in pbc]
    translations = np.array(list(itertools.product(*pbc_directions))).T
    p0r = np.tile(np.reshape(P[0, :], (3, 1)), (1, translations.shape[1]))
    p1r = np.tile(np.reshape(P[1, :], (3, 1)), (1, translations.shape[1]))
    dp_vec = p0r + np.dot(ct, translations)
    d = np.min(np.power(p1r - dp_vec, 2).sum(axis=0))**0.5
    return d


def db_call_with_error_tol(db_cursor, expression, args=[]):
    """In case the GA is used on older versions of networking
    filesystems there might be some delays. For this reason
    some extra error tolerance when calling the SQLite db is
    employed.
    """
    import sqlite3
    for _ in range(10):
        try:
            db_cursor.execute(expression, args)
            return
        except sqlite3.OperationalError as e:
            print(e)
            time.sleep(2.)
    raise sqlite3.OperationalError(
        'Database still locked after 10 attempts (20 s)')


def save_trajectory(confid, trajectory, folder):
    """Saves traj files to the database folder.
    This method should never be used directly,
    but only through the DataConnection object.
    """
    fname = os.path.join(folder, 'traj%05d.traj' % confid)
    write(fname, trajectory)
    return fname


def get_trajectory(fname):
    """Extra error tolerance when loading traj files."""
    fname = str(fname)
    try:
        t = read(fname)
    except OSError as e:
        print('get_trajectory error ' + e)
    return t


def gather_atoms_by_tag(atoms):
    """Translates same-tag atoms so that they lie 'together',
    with distance vectors as in the minimum image convention."""
    tags = atoms.get_tags()
    pos = atoms.get_positions()
    for tag in list(set(tags)):
        indices = np.where(tags == tag)[0]
        if len(indices) == 1:
            continue
        vectors = atoms.get_distances(indices[0], indices[1:],
                                      mic=True, vector=True)
        pos[indices[1:]] = pos[indices[0]] + vectors
    atoms.set_positions(pos)


def atoms_too_close(atoms, bl, use_tags=False):
    """Checks if any atoms in a are too close, as defined by
    the distances in the bl dictionary.

    use_tags: whether to use the Atoms tags to disable distance
        checking within a set of atoms with the same tag.

    Note: if certain atoms are constrained and use_tags is True,
    this method may return unexpected results in case the
    contraints prevent same-tag atoms to be gathered together in
    the minimum-image-convention. In such cases, one should
    (1) release the relevant constraints,
    (2) apply the gather_atoms_by_tag function, and
    (3) re-apply the constraints, before using the
        atoms_too_close function.
    """
    a = atoms.copy()
    if use_tags:
        gather_atoms_by_tag(a)

    pbc = a.get_pbc()
    cell = a.get_cell()
    num = a.get_atomic_numbers()
    pos = a.get_positions()
    tags = a.get_tags()
    unique_types = sorted(list(set(num)))

    neighbours = []
    for i in range(3):
        if pbc[i]:
            neighbours.append([-1, 0, 1])
        else:
            neighbours.append([0])

    for nx, ny, nz in itertools.product(*neighbours):
        displacement = np.dot(cell.T, np.array([nx, ny, nz]).T)
        pos_new = pos + displacement
        distances = cdist(pos, pos_new)

        if nx == 0 and ny == 0 and nz == 0:
            if use_tags and len(a) > 1:
                x = np.array([tags]).T
                distances += 1e2 * (cdist(x, x) == 0)
            else:
                distances += 1e2 * np.identity(len(a))

        iterator = itertools.combinations_with_replacement(unique_types, 2)
        for type1, type2 in iterator:
            x1 = np.where(num == type1)
            x2 = np.where(num == type2)
            if np.min(distances[x1].T[x2]) < bl[(type1, type2)]:
                return True

    return False


def atoms_too_close_two_sets(a, b, bl):
    """Checks if any atoms in a are too close to an atom in b,
    as defined by the bl dictionary."""
    pbc_a = a.get_pbc()
    pbc_b = b.get_pbc()
    cell_a = a.get_cell()
    cell_b = a.get_cell()
    assert np.allclose(pbc_a, pbc_b), (pbc_a, pbc_b)
    assert np.allclose(cell_a, cell_b), (cell_a, cell_b)

    pos_a = a.get_positions()
    pos_b = b.get_positions()

    num_a = a.get_atomic_numbers()
    num_b = b.get_atomic_numbers()
    unique_types = sorted(set(list(num_a) + list(num_b)))

    neighbours = []
    for i in range(3):
        neighbours.append([-1, 0, 1] if pbc_a[i] else [0])

    for nx, ny, nz in itertools.product(*neighbours):
        displacement = np.dot(cell_a.T, np.array([nx, ny, nz]).T)
        pos_b_disp = pos_b + displacement
        distances = cdist(pos_a, pos_b_disp)

        for type1 in unique_types:
            if type1 not in num_a:
                continue
            x1 = np.where(num_a == type1)

            for type2 in unique_types:
                if type2 not in num_b:
                    continue
                x2 = np.where(num_b == type2)
                if np.min(distances[x1].T[x2]) < bl[(type1, type2)]:
                    return True
    return False


def get_all_atom_types(slab, atom_numbers_to_optimize):
    """Utility method used to extract all unique atom types
    from the atoms object slab and the list of atomic numbers
    atom_numbers_to_optimize.
    """
    from_slab = list(set(slab.numbers))
    from_top = list(set(atom_numbers_to_optimize))
    from_slab.extend(from_top)
    return list(set(from_slab))


def get_distance_matrix(atoms, self_distance=1000):
    """NB: This function is way slower than atoms.get_all_distances()

    Returns a numpy matrix with the distances between the atoms
    in the supplied atoms object, with the indices of the matrix
    corresponding to the indices in the atoms object.

    The parameter self_distance will be put in the diagonal
    elements ([i][i])
    """
    dm = np.zeros([len(atoms), len(atoms)])
    for i in range(len(atoms)):
        dm[i][i] = self_distance
        for j in range(i + 1, len(atoms)):
            rij = atoms.get_distance(i, j)
            dm[i][j] = rij
            dm[j][i] = rij
    return dm


def get_nndist(atoms, distance_matrix):
    """Returns an estimate of the nearest neighbor bond distance
    in the supplied atoms object given the supplied distance_matrix.

    The estimate comes from the first peak in the radial distribution
    function.
    """
    rmax = 10.  # No bonds longer than 10 angstrom expected
    nbins = 200
    rdf, dists = get_rdf(atoms, rmax, nbins, distance_matrix)
    return dists[np.argmax(rdf)]


def get_nnmat(atoms, mic=False):
    """Calculate the nearest neighbor matrix as specified in
    S. Lysgaard et al., Top. Catal., 2014, 57 (1-4), pp 33-39

    Returns an array of average numbers of nearest neighbors
    the order is determined by self.elements.
    Example: self.elements = ["Cu", "Ni"]
    get_nnmat returns a single list [Cu-Cu bonds/N(Cu),
    Cu-Ni bonds/N(Cu), Ni-Cu bonds/N(Ni), Ni-Ni bonds/N(Ni)]
    where N(element) is the number of atoms of the type element
    in the atoms object.

    The distance matrix can be quite costly to calculate every
    time nnmat is required (and disk intensive if saved), thus
    it makes sense to calculate nnmat along with e.g. the
    potential energy and save it in atoms.info['data']['nnmat'].
    """
    if 'data' in atoms.info and 'nnmat' in atoms.info['data']:
        return atoms.info['data']['nnmat']
    elements = sorted(set(atoms.get_chemical_symbols()))
    nnmat = np.zeros((len(elements), len(elements)))
    # dm = get_distance_matrix(atoms)
    dm = atoms.get_all_distances(mic=mic)
    nndist = get_nndist(atoms, dm) + 0.2
    for i in range(len(atoms)):
        row = [j for j in range(len(elements))
               if atoms[i].symbol == elements[j]][0]
        neighbors = [j for j in range(len(dm[i])) if dm[i][j] < nndist]
        for n in neighbors:
            column = [j for j in range(len(elements))
                      if atoms[n].symbol == elements[j]][0]
            nnmat[row][column] += 1
    # divide by the number of that type of atoms in the structure
    for i, el in enumerate(elements):
        nnmat[i] /= len([j for j in range(len(atoms))
                         if atoms[int(j)].symbol == el])
    # makes a single list out of a list of lists
    nnlist = np.reshape(nnmat, (len(nnmat)**2))
    return nnlist


def get_nnmat_string(atoms, decimals=2, mic=False):
    nnmat = get_nnmat(atoms, mic=mic)
    s = '-'.join(['{1:2.{0}f}'.format(decimals, i)
                  for i in nnmat])
    if len(nnmat) == 1:
        return s + '-'
    return s


def get_connections_index(atoms, max_conn=5, no_count_types=None):
    """This method returns a dictionary where each key value are a
    specific number of neighbors and list of atoms indices with
    that amount of neighbors respectively. The method utilizes the
    neighbor list and hence inherit the restrictions for
    neighbors. Option added to remove connections between
    defined atom types.

    Parameters
    ----------

    atoms : Atoms object
        The connections will be counted using this supplied Atoms object

    max_conn : int
        Any atom with more connections than this will be counted as
        having max_conn connections.
        Default 5

    no_count_types : list or None
        List of atomic numbers that should be excluded in the count.
        Default None (meaning all atoms count).
    """
    conn = get_neighbor_list(atoms)

    if conn is None:
        conn = get_neighborlist(atoms)

    if no_count_types is None:
        no_count_types = []

    conn_index = {}
    for i in range(len(atoms)):
        if atoms[i].number not in no_count_types:
            cconn = min(len(conn[i]), max_conn - 1)
            if cconn not in conn_index:
                conn_index[cconn] = []
            conn_index[cconn].append(i)

    return conn_index


def get_atoms_connections(atoms, max_conn=5, no_count_types=None):
    """This method returns a list of the numbers of atoms
    with X number of neighbors. The method utilizes the
    neighbor list and hence inherit the restrictions for
    neighbors. Option added to remove connections between
    defined atom types.
    """
    conn_index = get_connections_index(atoms, max_conn=max_conn,
                                       no_count_types=no_count_types)

    no_of_conn = [0] * max_conn
    for i in conn_index:
        no_of_conn[i] += len(conn_index[i])

    return no_of_conn


def get_angles_distribution(atoms, ang_grid=9):
    """Method to get the distribution of bond angles
    in bins (default 9) with bonds defined from
    the get_neighbor_list().
    """
    conn = get_neighbor_list(atoms)

    if conn is None:
        conn = get_neighborlist(atoms)

    bins = [0] * ang_grid

    for atom in atoms:
        for i in conn[atom.index]:
            for j in conn[atom.index]:
                if j != i:
                    a = atoms.get_angle(i, atom.index, j)
                    for k in range(ang_grid):
                        if (k + 1) * 180. / ang_grid > a > k * 180. / ang_grid:
                            bins[k] += 1
    # Removing dobbelt counting
    for i in range(ang_grid):
        bins[i] /= 2.
    return bins


def get_neighborlist(atoms, dx=0.2, no_count_types=None):
    """Method to get the a dict with list of neighboring
    atoms defined as the two covalent radii + fixed distance.
    Option added to remove neighbors between defined atom types.
    """
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    if no_count_types is None:
        no_count_types = []

    conn = {}
    for atomi in atoms:
        conn_this_atom = []
        for atomj in atoms:
            if atomi.index != atomj.index:
                if atomi.number not in no_count_types:
                    if atomj.number not in no_count_types:
                        d = get_mic_distance(atomi.position,
                                             atomj.position,
                                             cell,
                                             pbc)
                        cri = covalent_radii[atomi.number]
                        crj = covalent_radii[atomj.number]
                        d_max = crj + cri + dx
                        if d < d_max:
                            conn_this_atom.append(atomj.index)
        conn[atomi.index] = conn_this_atom
    return conn


def get_atoms_distribution(atoms, number_of_bins=5, max_distance=8,
                           center=None, no_count_types=None):
    """Method to get the distribution of atoms in the
    structure in bins of distances from a defined
    center. Option added to remove counting of
    certain atom types.
    """
    pbc = atoms.get_pbc()
    cell = atoms.get_cell()
    if center is None:
        # Center used for the atom distribution if None is supplied!
        cx = sum(cell[:, 0]) / 2.
        cy = sum(cell[:, 1]) / 2.
        cz = sum(cell[:, 2]) / 2.
        center = (cx, cy, cz)
    bins = [0] * number_of_bins

    if no_count_types is None:
        no_count_types = []

    for atom in atoms:
        if atom.number not in no_count_types:
            d = get_mic_distance(atom.position, center, cell, pbc)
            for k in range(number_of_bins - 1):
                min_dis_cur_bin = k * max_distance / (number_of_bins - 1.)
                max_dis_cur_bin = ((k + 1) * max_distance /
                                   (number_of_bins - 1.))
                if min_dis_cur_bin < d < max_dis_cur_bin:
                    bins[k] += 1
            if d > max_distance:
                bins[number_of_bins - 1] += 1
    return bins


def get_rings(atoms, rings=[5, 6, 7]):
    """This method return a list of the number of atoms involved
    in rings in the structures. It uses the neighbor
    list hence inherit the restriction used for neighbors.
    """
    conn = get_neighbor_list(atoms)

    if conn is None:
        conn = get_neighborlist(atoms)

    no_of_loops = [0] * 8
    for s1 in range(len(atoms)):
        for s2 in conn[s1]:
            v12 = [s1] + [s2]
            for s3 in [s for s in conn[s2] if s not in v12]:
                v13 = v12 + [s3]
                if s1 in conn[s3]:
                    no_of_loops[3] += 1
                for s4 in [s for s in conn[s3] if s not in v13]:
                    v14 = v13 + [s4]
                    if s1 in conn[s4]:
                        no_of_loops[4] += 1
                    for s5 in [s for s in conn[s4] if s not in v14]:
                        v15 = v14 + [s5]
                        if s1 in conn[s5]:
                            no_of_loops[5] += 1
                        for s6 in [s for s in conn[s5] if s not in v15]:
                            v16 = v15 + [s6]
                            if s1 in conn[s6]:
                                no_of_loops[6] += 1
                            for s7 in [s for s in conn[s6] if s not in v16]:
                                # v17 = v16 + [s7]
                                if s1 in conn[s7]:
                                    no_of_loops[7] += 1

    to_return = []
    for ring in rings:
        to_return.append(no_of_loops[ring])

    return to_return


def get_cell_angles_lengths(cell):
    """Returns cell vectors lengths (a,b,c) as well as different
    angles (alpha, beta, gamma, phi, chi, psi) (in radians).
    """
    cellpar = cell_to_cellpar(cell)
    cellpar[3:] *= np.pi / 180  # convert angles to radians
    parnames = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
    values = {n: p for n, p in zip(parnames, cellpar)}

    volume = abs(np.linalg.det(cell))
    for i, param in enumerate(['phi', 'chi', 'psi']):
        ab = np.linalg.norm(
            np.cross(cell[(i + 1) % 3, :], cell[(i + 2) % 3, :]))
        c = np.linalg.norm(cell[i, :])
        s = np.abs(volume / (ab * c))
        if 1 + 1e-6 > s > 1:
            s = 1.
        values[param] = np.arcsin(s)

    return values


class CellBounds:
    """Class for defining as well as checking limits on
    cell vector lengths and angles.

    Parameters:

    bounds: dict
        Any of the following keywords can be used, in
        conjunction with a [low, high] list determining
        the lower and upper bounds:

        a, b, c:
           Minimal and maximal lengths (in Angstrom)
           for the 1st, 2nd and 3rd lattice vectors.

        alpha, beta, gamma:
           Minimal and maximal values (in degrees)
           for the angles between the lattice vectors.

        phi, chi, psi:
           Minimal and maximal values (in degrees)
           for the angles between each lattice vector
           and the plane defined by the other two vectors.

    Example:

    >>> from ase.ga.utilities import CellBounds
    >>> CellBounds(bounds={'phi': [20, 160],
    ...                    'chi': [60, 120],
    ...                    'psi': [20, 160],
    ...                    'a': [2, 20], 'b': [2, 20], 'c': [2, 20]})
    """

    def __init__(self, bounds={}):
        self.bounds = {'alpha': [0, np.pi], 'beta': [0, np.pi],
                       'gamma': [0, np.pi], 'phi': [0, np.pi],
                       'chi': [0, np.pi], 'psi': [0, np.pi],
                       'a': [0, 1e6], 'b': [0, 1e6], 'c': [0, 1e6]}

        for param, bound in bounds.items():
            if param not in ['a', 'b', 'c']:
                # Convert angle from degree to radians
                bound = [b * np.pi / 180. for b in bound]
            self.bounds[param] = bound

    def is_within_bounds(self, cell):
        values = get_cell_angles_lengths(cell)
        verdict = True
        for param, bound in self.bounds.items():
            if not (bound[0] <= values[param] <= bound[1]):
                verdict = False
        return verdict


def get_rotation_matrix(u, t):
    """Returns the transformation matrix for rotation over
    an angle t along an axis with direction u.
    """
    ux, uy, uz = u
    cost, sint = np.cos(t), np.sin(t)
    rotmat = np.array([[(ux**2) * (1 - cost) + cost,
                        ux * uy * (1 - cost) - uz * sint,
                        ux * uz * (1 - cost) + uy * sint],
                       [ux * uy * (1 - cost) + uz * sint,
                        (uy**2) * (1 - cost) + cost,
                        uy * uz * (1 - cost) - ux * sint],
                       [ux * uz * (1 - cost) - uy * sint,
                        uy * uz * (1 - cost) + ux * sint,
                        (uz**2) * (1 - cost) + cost]])
    return rotmat
