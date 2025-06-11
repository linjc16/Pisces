# fmt: off

"""
Implementation of the Precon abstract base class and subclasses
"""

import copy
import sys
import time
import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import spsolve

import ase.units as units
import ase.utils.ff as ff
from ase.constraints import FixAtoms
from ase.filters import Filter
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list
from ase.optimize.precon.neighbors import (
    estimate_nearest_neighbour_distance,
    get_neighbours,
)
from ase.utils import longsum, tokenize_version

try:
    import pyamg
except ImportError:
    have_pyamg = False
else:
    have_pyamg = True


def create_pyamg_solver(P, max_levels=15):
    if not have_pyamg:
        raise RuntimeError(
            'pyamg not available: install with `pip install pyamg`')
    filter_key = 'filter'
    if tokenize_version(pyamg.__version__) >= tokenize_version('4.2.1'):
        filter_key = 'filter_entries'
    return pyamg.smoothed_aggregation_solver(
        P, B=None,
        strength=('symmetric', {'theta': 0.0}),
        smooth=(
            'jacobi', {filter_key: True, 'weighting': 'local'}),
        improve_candidates=([('block_gauss_seidel',
                              {'sweep': 'symmetric', 'iterations': 4})] +
                            [None] * (max_levels - 1)),
        aggregate='standard',
        presmoother=('block_gauss_seidel',
                     {'sweep': 'symmetric', 'iterations': 1}),
        postsmoother=('block_gauss_seidel',
                      {'sweep': 'symmetric', 'iterations': 1}),
        max_levels=max_levels,
        max_coarse=300,
        coarse_solver='pinv')


THz = 1e12 * 1. / units.s


class Precon(ABC):

    @abstractmethod
    def make_precon(self, atoms, reinitialize=None):
        """
        Create a preconditioner matrix based on the passed set of atoms.

        Creates a general-purpose preconditioner for use with optimization
        algorithms, based on examining distances between pairs of atoms in the
        lattice. The matrix will be stored in the attribute self.P and
        returned.

        Args:
            atoms: the Atoms object used to create the preconditioner.

            reinitialize: if True, parameters of the preconditioner
                will be recalculated before the preconditioner matrix is
                created. If False, they will be calculated only when they
                do not currently have a value (ie, the first time this
                function is called).

        Returns:
            P: A sparse scipy csr_matrix. BE AWARE that using
                numpy.dot() with sparse matrices will result in
                errors/incorrect results - use the .dot method directly
                on the matrix instead.
        """
        ...

    @abstractmethod
    def Pdot(self, x):
        """
        Return the result of applying P to a vector x
        """
        ...

    def dot(self, x, y):
        """
        Return the preconditioned dot product <P x, y>

        Uses 128-bit floating point math for vector dot products
        """
        return longsum(self.Pdot(x) * y)

    def norm(self, x):
        """
        Return the P-norm of x, where |x|_P = sqrt(<Px, x>)
        """
        return np.sqrt(self.dot(x, x))

    def vdot(self, x, y):
        return self.dot(x.reshape(-1),
                        y.reshape(-1))

    @abstractmethod
    def solve(self, x):
        """
        Solve the (sparse) linear system P x = y and return y
        """
        ...

    def apply(self, forces, atoms):
        """
        Convenience wrapper that combines make_precon() and solve()

        Parameters
        ----------
        forces: array
            (len(atoms)*3) array of input forces
        atoms: ase.atoms.Atoms

        Returns
        -------
        precon_forces: array
            (len(atoms), 3) array of preconditioned forces
        residual: float
            inf-norm of original forces, i.e. maximum absolute force
        """
        self.make_precon(atoms)
        residual = np.linalg.norm(forces, np.inf)
        precon_forces = self.solve(forces)
        return precon_forces, residual

    @abstractmethod
    def copy(self):
        ...

    @abstractmethod
    def asarray(self):
        """
        Array representation of preconditioner, as a dense matrix
        """
        ...


class Logfile:
    def __init__(self, logfile=None):
        if isinstance(logfile, str):
            if logfile == "-":
                logfile = sys.stdout
            else:
                logfile = open(logfile, "a")
        self.logfile = logfile

    def write(self, *args):
        if self.logfile is None:
            return
        self.logfile.write(*args)


class SparsePrecon(Precon):
    def __init__(self, r_cut=None, r_NN=None,
                 mu=None, mu_c=None,
                 dim=3, c_stab=0.1, force_stab=False,
                 reinitialize=False, array_convention='C',
                 solver="auto", solve_tol=1e-8,
                 apply_positions=True, apply_cell=True,
                 estimate_mu_eigmode=False, logfile=None, rng=None,
                 neighbour_list=neighbor_list):
        """Initialise a preconditioner object based on passed parameters.

        Parameters:
            r_cut: float. This is a cut-off radius. The preconditioner matrix
                will be created by considering pairs of atoms that are within a
                distance r_cut of each other. For a regular lattice, this is
                usually taken somewhere between the first- and second-nearest
                neighbour distance. If r_cut is not provided, default is
                2 * r_NN (see below)
            r_NN: nearest neighbour distance. If not provided, this is
                  calculated
                from input structure.
            mu: float
                energy scale for position degreees of freedom. If `None`, mu
                is precomputed using finite difference derivatives.
            mu_c: float
                energy scale for cell degreees of freedom. Also precomputed
                if None.
            estimate_mu_eigmode:
                If True, estimates mu based on the lowest eigenmodes of
                unstabilised preconditioner. If False it uses the sine based
                approach.
            dim: int; dimensions of the problem
            c_stab: float. The diagonal of the preconditioner matrix will have
                a stabilisation constant added, which will be the value of
                c_stab times mu.
            force_stab:
                If True, always add the stabilisation to diagnonal, regardless
                of the presence of fixed atoms.
            reinitialize: if True, the value of mu will be recalculated when
                self.make_precon is called. This can be overridden in specific
                cases with reinitialize argument in self.make_precon. If it
                is set to True here, the value passed for mu will be
                irrelevant unless reinitialize is set to False the first time
                make_precon is called.
            array_convention: Either 'C' or 'F' for Fortran; this will change
                the preconditioner to reflect the ordering of the indices in
                the vector it will operate on. The C convention assumes the
                vector will be arranged atom-by-atom (ie [x1, y1, z1, x2, ...])
                while the F convention assumes it will be arranged component
                by component (ie [x1, x2, ..., y1, y2, ...]).
            solver: One of "auto", "direct" or "pyamg", specifying whether to
                use a direct sparse solver or PyAMG to solve P x = y.
                Default is "auto" which uses PyAMG if available, falling
                back to sparse solver if not. solve_tol: tolerance used for
                PyAMG sparse linear solver, if available.
            apply_positions:  bool
                if True, apply preconditioner to position DoF
            apply_cell: bool
                if True, apply preconditioner to cell DoF
            logfile: file object or str
                If *logfile* is a string, a file with that name will be opened.
                Use '-' for stdout.
            rng: None or np.random.RandomState instance
                Random number generator to use for initialising pyamg solver
            neighbor_list: function (optional). Optionally replace the built-in
                ASE neighbour list with an alternative with the same call
                signature, e.g. `matscipy.neighbours.neighbour_list`.

        Raises:
            ValueError for problem with arguments

        """
        self.r_NN = r_NN
        self.r_cut = r_cut
        self.mu = mu
        self.mu_c = mu_c
        self.estimate_mu_eigmode = estimate_mu_eigmode
        self.c_stab = c_stab
        self.force_stab = force_stab
        self.array_convention = array_convention
        self.reinitialize = reinitialize
        self.P = None
        self.old_positions = None

        use_pyamg = False
        if solver == "auto":
            use_pyamg = have_pyamg
        elif solver == "direct":
            use_pyamg = False
        elif solver == "pyamg":
            if not have_pyamg:
                raise RuntimeError("solver='pyamg', PyAMG can't be imported!")
            use_pyamg = True
        else:
            raise ValueError('unknown solver - '
                             'should be "auto", "direct" or "pyamg"')

        self.use_pyamg = use_pyamg
        self.solve_tol = solve_tol
        self.apply_positions = apply_positions
        self.apply_cell = apply_cell

        if dim < 1:
            raise ValueError('Dimension must be at least 1')
        self.dim = dim
        self.logfile = Logfile(logfile)

        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        self.neighbor_list = neighbor_list

    def copy(self):
        return copy.deepcopy(self)

    def Pdot(self, x):
        return self.P.dot(x)

    def solve(self, x):
        start_time = time.time()
        if self.use_pyamg and have_pyamg:
            y = self.ml.solve(x, x0=self.rng.random(self.P.shape[0]),
                              tol=self.solve_tol,
                              accel='cg',
                              maxiter=300,
                              cycle='W')
        else:
            y = spsolve(self.P, x)
        self.logfile.write(
            f'--- Precon applied in {(time.time() - start_time)} seconds ---\n')
        return y

    def estimate_mu(self, atoms, H=None):
        r"""Estimate optimal preconditioner coefficient \mu

        \mu is estimated from a numerical solution of

            [dE(p+v) -  dE(p)] \cdot v = \mu < P1 v, v >

        with perturbation

            v(x,y,z) = H P_lowest_nonzero_eigvec(x, y, z)

            or

            v(x,y,z) = H (sin(x / Lx), sin(y / Ly), sin(z / Lz))

        After the optimal \mu is found, self.mu will be set to its value.

        If `atoms` is an instance of Filter an additional \mu_c
        will be computed for the cell degrees of freedom .

        Args:
            atoms: Atoms object for initial system

            H: 3x3 array or None
                Magnitude of deformation to apply.
                Default is 1e-2*rNN*np.eye(3)

        Returns:
            mu   : float
            mu_c : float or None
        """
        logfile = self.logfile

        if self.dim != 3:
            raise ValueError('Automatic calculation of mu only possible for '
                             'three-dimensional preconditioners. Try setting '
                             'mu manually instead.')

        if self.r_NN is None:
            self.r_NN = estimate_nearest_neighbour_distance(atoms,
                                                            self.neighbor_list)

        # deformation matrix, default is diagonal
        if H is None:
            H = 1e-2 * self.r_NN * np.eye(3)

        # compute perturbation
        p = atoms.get_positions()

        if self.estimate_mu_eigmode:
            self.mu = 1.0
            self.mu_c = 1.0
            c_stab = self.c_stab
            self.c_stab = 0.0

            if isinstance(atoms, Filter):
                n = len(atoms.atoms)
            else:
                n = len(atoms)
            self._make_sparse_precon(atoms, initial_assembly=True)
            self.P = self.P[:3 * n, :3 * n]
            eigvals, eigvecs = sparse.linalg.eigsh(self.P, k=4, which='SM')

            logfile.write('estimate_mu(): lowest 4 eigvals = %f %f %f %f\n' %
                          (eigvals[0], eigvals[1], eigvals[2], eigvals[3]))
            # check eigenvalues
            if any(eigvals[0:3] > 1e-6):
                raise ValueError('First 3 eigenvalues of preconditioner matrix'
                                 'do not correspond to translational modes.')
            elif eigvals[3] < 1e-6:
                raise ValueError('Fourth smallest eigenvalue of '
                                 'preconditioner matrix '
                                 'is too small, increase r_cut.')

            x = np.zeros(n)
            for i in range(n):
                x[i] = eigvecs[:, 3][3 * i]
            x = x / np.linalg.norm(x)
            if x[0] < 0:
                x = -x

            v = np.zeros(3 * len(atoms))
            for i in range(n):
                v[3 * i] = x[i]
                v[3 * i + 1] = x[i]
                v[3 * i + 2] = x[i]
            v = v / np.linalg.norm(v)
            v = v.reshape((-1, 3))

            self.c_stab = c_stab
        else:
            Lx, Ly, Lz = (p[:, i].max() - p[:, i].min() for i in range(3))
            logfile.write('estimate_mu(): Lx=%.1f Ly=%.1f Lz=%.1f\n' %
                          (Lx, Ly, Lz))

            x, y, z = p.T
            # sine_vr = [np.sin(x/Lx), np.sin(y/Ly), np.sin(z/Lz)], but we need
            # to take into account the possibility that one of Lx/Ly/Lz is
            # zero.
            sine_vr = [x, y, z]

            for i, L in enumerate([Lx, Ly, Lz]):
                if L == 0:
                    warnings.warn(
                        'Cell length L[%d] == 0. Setting H[%d,%d] = 0.' %
                        (i, i, i))
                    H[i, i] = 0.0
                else:
                    sine_vr[i] = np.sin(sine_vr[i] / L)

            v = np.dot(H, sine_vr).T

        natoms = len(atoms)
        if isinstance(atoms, Filter):
            natoms = len(atoms.atoms)
            eps = H / self.r_NN
            v[natoms:, :] = eps

        v1 = v.reshape(-1)

        # compute LHS
        dE_p = -atoms.get_forces().reshape(-1)
        atoms_v = atoms.copy()
        atoms_v.calc = atoms.calc
        if isinstance(atoms, Filter):
            atoms_v = atoms.__class__(atoms_v)
            if hasattr(atoms, 'constant_volume'):
                atoms_v.constant_volume = atoms.constant_volume
        atoms_v.set_positions(p + v)
        dE_p_plus_v = -atoms_v.get_forces().reshape(-1)

        # compute left hand side
        LHS = (dE_p_plus_v - dE_p) * v1

        # assemble P with \mu = 1
        self.mu = 1.0
        self.mu_c = 1.0

        self._make_sparse_precon(atoms, initial_assembly=True)

        # compute right hand side
        RHS = self.P.dot(v1) * v1

        # use partial sums to compute separate mu for positions and cell DoFs
        self.mu = longsum(LHS[:3 * natoms]) / longsum(RHS[:3 * natoms])
        if self.mu < 1.0:
            logfile.write('estimate_mu(): mu (%.3f) < 1.0, '
                          'capping at mu=1.0' % self.mu)
            self.mu = 1.0

        if isinstance(atoms, Filter):
            self.mu_c = longsum(LHS[3 * natoms:]) / longsum(RHS[3 * natoms:])
            if self.mu_c < 1.0:
                logfile.write('estimate_mu(): mu_c (%.3f) < 1.0, '
                              'capping at mu_c=1.0\n' % self.mu_c)
                self.mu_c = 1.0

        logfile.write('estimate_mu(): mu=%r, mu_c=%r\n' %
                      (self.mu, self.mu_c))

        self.P = None  # force a rebuild with new mu (there may be fixed atoms)
        return (self.mu, self.mu_c)

    def asarray(self):
        return np.array(self.P.todense())

    def one_dim_to_ndim(self, csc_P, N):
        """
        Expand an N x N precon matrix to self.dim*N x self.dim * N

        Args:
            csc_P (sparse matrix): N x N sparse matrix, in CSC format
        """
        start_time = time.time()
        if self.dim == 1:
            P = csc_P
        elif self.array_convention == 'F':
            csc_P = csc_P.tocsr()
            P = csc_P
            for _ in range(self.dim - 1):
                P = sparse.block_diag((P, csc_P)).tocsr()
        else:
            # convert back to triplet and read the arrays
            csc_P = csc_P.tocoo()
            i = csc_P.row * self.dim
            j = csc_P.col * self.dim
            z = csc_P.data

            # N-dimensionalise, interlaced coordinates
            I = np.hstack([i + d for d in range(self.dim)])
            J = np.hstack([j + d for d in range(self.dim)])
            Z = np.hstack([z for _ in range(self.dim)])
            P = sparse.csc_matrix((Z, (I, J)),
                                  shape=(self.dim * N, self.dim * N))
            P = P.tocsr()
        self.logfile.write(
            f'--- N-dim precon created in {(time.time() - start_time)} s ---\n')
        return P

    def create_solver(self):
        if self.use_pyamg and have_pyamg:
            start_time = time.time()
            self.ml = create_pyamg_solver(self.P)
            self.logfile.write(
                f'--- multi grid solver created in {(time.time() - start_time)}'
                ' ---\n')


class SparseCoeffPrecon(SparsePrecon):
    def _make_sparse_precon(self, atoms, initial_assembly=False,
                            force_stab=False):
        """Create a sparse preconditioner matrix based on the passed atoms.

        Creates a general-purpose preconditioner for use with optimization
        algorithms, based on examining distances between pairs of atoms in the
        lattice. The matrix will be stored in the attribute self.P and
        returned. Note that this function will use self.mu, whatever it is.

        Args:
            atoms: the Atoms object used to create the preconditioner.

        Returns:
            A scipy.sparse.csr_matrix object, representing a d*N by d*N matrix
            (where N is the number of atoms, and d is the value of self.dim).
            BE AWARE that using numpy.dot() with this object will result in
            errors/incorrect results - use the .dot method directly on the
            sparse matrix instead.

        """
        logfile = self.logfile
        logfile.write('creating sparse precon: initial_assembly=%r, '
                      'force_stab=%r, apply_positions=%r, apply_cell=%r\n' %
                      (initial_assembly, force_stab, self.apply_positions,
                       self.apply_cell))

        N = len(atoms)
        diag_i = np.arange(N, dtype=int)
        start_time = time.time()
        if self.apply_positions:
            # compute neighbour list
            i, j, rij, fixed_atoms = get_neighbours(
                atoms, self.r_cut,
                neighbor_list=self.neighbor_list)
            logfile.write(
                f'--- neighbour list created in {(time.time() - start_time)} s '
                '--- \n')

            # compute entries in triplet format: without the constraints
            start_time = time.time()
            coeff = self.get_coeff(rij)
            diag_coeff = np.bincount(i, -coeff, minlength=N).astype(np.float64)
            if force_stab or len(fixed_atoms) == 0:
                logfile.write('adding stabilisation to precon')
                diag_coeff += self.mu * self.c_stab
        else:
            diag_coeff = np.ones(N)

        # precon is mu_c * identity for cell DoF
        if isinstance(atoms, Filter):
            if self.apply_cell:
                diag_coeff[-3:] = self.mu_c
            else:
                diag_coeff[-3:] = 1.0
        logfile.write(
            f'--- computed triplet format in {(time.time() - start_time)} s '
            '---\n')

        if self.apply_positions and not initial_assembly:
            # apply the constraints
            start_time = time.time()
            mask = np.ones(N)
            mask[fixed_atoms] = 0.0
            coeff *= mask[i] * mask[j]
            diag_coeff[fixed_atoms] = 1.0
            logfile.write(
                f'--- applied fixed_atoms in {(time.time() - start_time)} s '
                '---\n')

        if self.apply_positions:
            # remove zeros
            start_time = time.time()
            inz = np.nonzero(coeff)
            i = np.hstack((i[inz], diag_i))
            j = np.hstack((j[inz], diag_i))
            coeff = np.hstack((coeff[inz], diag_coeff))
            logfile.write(
                f'--- remove zeros in {(time.time() - start_time)} s '
                '---\n')
        else:
            i = diag_i
            j = diag_i
            coeff = diag_coeff

        # create an N x N precon matrix in compressed sparse column (CSC) format
        start_time = time.time()
        csc_P = sparse.csc_matrix((coeff, (i, j)), shape=(N, N))
        logfile.write(
            f'--- created CSC matrix in {(time.time() - start_time)} s '
            '---\n')

        self.P = self.one_dim_to_ndim(csc_P, N)
        self.create_solver()

    def make_precon(self, atoms, reinitialize=None):
        if self.r_NN is None:
            self.r_NN = estimate_nearest_neighbour_distance(atoms,
                                                            self.neighbor_list)

        if self.r_cut is None:
            # This is the first time this function has been called, and no
            # cutoff radius has been specified, so calculate it automatically.
            self.r_cut = 2.0 * self.r_NN
        elif self.r_cut < self.r_NN:
            warning = ('WARNING: r_cut (%.2f) < r_NN (%.2f), '
                       'increasing to 1.1*r_NN = %.2f' % (self.r_cut,
                                                          self.r_NN,
                                                          1.1 * self.r_NN))
            warnings.warn(warning)
            self.r_cut = 1.1 * self.r_NN

        if reinitialize is None:
            # The caller has not specified whether or not to recalculate mu,
            # so the Precon's setting is used.
            reinitialize = self.reinitialize

        if self.mu is None:
            # Regardless of what the caller has specified, if we don't
            # currently have a value of mu, then we need one.
            reinitialize = True

        if reinitialize:
            self.estimate_mu(atoms)

        if self.P is not None:
            real_atoms = atoms
            if isinstance(atoms, Filter):
                real_atoms = atoms.atoms
            if self.old_positions is None:
                self.old_positions = real_atoms.positions
            displacement, _ = find_mic(real_atoms.positions -
                                       self.old_positions,
                                       real_atoms.cell, real_atoms.pbc)
            self.old_positions = real_atoms.get_positions()
            max_abs_displacement = abs(displacement).max()
            self.logfile.write('max(abs(displacements)) = %.2f A (%.2f r_NN)' %
                               (max_abs_displacement,
                                max_abs_displacement / self.r_NN))
            if max_abs_displacement < 0.5 * self.r_NN:
                return

        start_time = time.time()
        self._make_sparse_precon(atoms, force_stab=self.force_stab)
        self.logfile.write(
            f'--- Precon created in {(time.time() - start_time)} seconds '
            '--- \n')

    @abstractmethod
    def get_coeff(self, r):
        ...


class Pfrommer(Precon):
    """
    Use initial guess for inverse Hessian from Pfrommer et al. as a
    simple preconditioner

    J. Comput. Phys. vol 131 p233-240 (1997)
    """

    def __init__(self, bulk_modulus=500 * units.GPa, phonon_frequency=50 * THz,
                 apply_positions=True, apply_cell=True):
        """
        Default bulk modulus is 500 GPa and default phonon frequency is 50 THz
        """
        self.bulk_modulus = bulk_modulus
        self.phonon_frequency = phonon_frequency
        self.apply_positions = apply_positions
        self.apply_cell = apply_cell
        self.H0 = None

    def make_precon(self, atoms, reinitialize=None):
        if self.H0 is not None:
            # only build H0 on first call
            return

        variable_cell = False
        if isinstance(atoms, Filter):
            variable_cell = True
            atoms = atoms.atoms

        # position DoF
        omega = self.phonon_frequency
        mass = atoms.get_masses().mean()
        block = np.eye(3) / (mass * omega**2)
        blocks = [block] * len(atoms)

        # cell DoF
        if variable_cell:
            coeff = 1.0
            if self.apply_cell:
                coeff = 1.0 / (3 * self.bulk_modulus)
            blocks.append(np.diag([coeff] * 9))

        self.H0 = sparse.block_diag(blocks, format='csr')
        return

    def Pdot(self, x):
        return self.H0.solve(x)

    def solve(self, x):
        y = self.H0.dot(x)
        return y

    def copy(self):
        return Pfrommer(self.bulk_modulus,
                        self.phonon_frequency,
                        self.apply_positions,
                        self.apply_cell)

    def asarray(self):
        return np.array(self.H0.todense())


class IdentityPrecon(Precon):
    """
    Dummy preconditioner which does not modify forces
    """

    def make_precon(self, atoms, reinitialize=None):
        self.atoms = atoms

    def Pdot(self, x):
        return x

    def solve(self, x):
        return x

    def copy(self):
        return IdentityPrecon()

    def asarray(self):
        return np.eye(3 * len(self.atoms))


class C1(SparseCoeffPrecon):
    """Creates matrix by inserting a constant whenever r_ij is less than r_cut.
    """

    def __init__(self, r_cut=None, mu=None, mu_c=None, dim=3, c_stab=0.1,
                 force_stab=False,
                 reinitialize=False, array_convention='C',
                 solver="auto", solve_tol=1e-9,
                 apply_positions=True, apply_cell=True, logfile=None):
        super().__init__(r_cut=r_cut, mu=mu, mu_c=mu_c,
                         dim=dim, c_stab=c_stab,
                         force_stab=force_stab,
                         reinitialize=reinitialize,
                         array_convention=array_convention,
                         solver=solver, solve_tol=solve_tol,
                         apply_positions=apply_positions,
                         apply_cell=apply_cell,
                         logfile=logfile)

    def get_coeff(self, r):
        return -self.mu * np.ones_like(r)


class Exp(SparseCoeffPrecon):
    """Creates matrix with values decreasing exponentially with distance.
    """

    def __init__(self, A=3.0, r_cut=None, r_NN=None, mu=None, mu_c=None,
                 dim=3, c_stab=0.1,
                 force_stab=False, reinitialize=False, array_convention='C',
                 solver="auto", solve_tol=1e-9,
                 apply_positions=True, apply_cell=True,
                 estimate_mu_eigmode=False, logfile=None):
        """
        Initialise an Exp preconditioner with given parameters.

        Args:
            r_cut, mu, c_stab, dim, sparse, reinitialize, array_convention: see
                precon.__init__()
            A: coefficient in exp(-A*r/r_NN). Default is A=3.0.
        """
        super().__init__(r_cut=r_cut, r_NN=r_NN,
                         mu=mu, mu_c=mu_c, dim=dim, c_stab=c_stab,
                         force_stab=force_stab,
                         reinitialize=reinitialize,
                         array_convention=array_convention,
                         solver=solver,
                         solve_tol=solve_tol,
                         apply_positions=apply_positions,
                         apply_cell=apply_cell,
                         estimate_mu_eigmode=estimate_mu_eigmode,
                         logfile=logfile)

        self.A = A

    def get_coeff(self, r):
        return -self.mu * np.exp(-self.A * (r / self.r_NN - 1))


def ij_to_x(i, j):
    x = [3 * i, 3 * i + 1, 3 * i + 2,
         3 * j, 3 * j + 1, 3 * j + 2]
    return x


def ijk_to_x(i, j, k):
    x = [3 * i, 3 * i + 1, 3 * i + 2,
         3 * j, 3 * j + 1, 3 * j + 2,
         3 * k, 3 * k + 1, 3 * k + 2]
    return x


def ijkl_to_x(i, j, k, l):
    x = [3 * i, 3 * i + 1, 3 * i + 2,
         3 * j, 3 * j + 1, 3 * j + 2,
         3 * k, 3 * k + 1, 3 * k + 2,
         3 * l, 3 * l + 1, 3 * l + 2]
    return x


def apply_fixed(atoms, P):
    fixed_atoms = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed_atoms.extend(list(constraint.index))
        else:
            raise TypeError(
                'only FixAtoms constraints are supported by Precon class')
    if len(fixed_atoms) != 0:
        P = P.tolil()
    for i in fixed_atoms:
        P[i, :] = 0.0
        P[:, i] = 0.0
        P[i, i] = 1.0
    return P


class FF(SparsePrecon):
    """Creates matrix using morse/bond/angle/dihedral force field parameters.
    """

    def __init__(self, dim=3, c_stab=0.1, force_stab=False,
                 array_convention='C', solver="auto", solve_tol=1e-9,
                 apply_positions=True, apply_cell=True,
                 hessian='spectral', morses=None, bonds=None, angles=None,
                 dihedrals=None, logfile=None):
        """Initialise an FF preconditioner with given parameters.

        Args:
             dim, c_stab, force_stab, array_convention, use_pyamg, solve_tol:
                see SparsePrecon.__init__()
             morses: Morse instance
             bonds: Bond instance
             angles: Angle instance
             dihedrals: Dihedral instance
        """

        if (morses is None and bonds is None and angles is None and
                dihedrals is None):
            raise ImportError(
                'At least one of morses, bonds, angles or dihedrals must be '
                'defined!')

        super().__init__(dim=dim, c_stab=c_stab,
                         force_stab=force_stab,
                         array_convention=array_convention,
                         solver=solver,
                         solve_tol=solve_tol,
                         apply_positions=apply_positions,
                         apply_cell=apply_cell, logfile=logfile)

        self.hessian = hessian
        self.morses = morses
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals

    def make_precon(self, atoms, reinitialize=None):
        start_time = time.time()
        self._make_sparse_precon(atoms, force_stab=self.force_stab)
        self.logfile.write(
            f'--- Precon created in {(time.time() - start_time)} seconds '
            '---\n')

    def add_morse(self, morse, atoms, row, col, data, conn=None):
        if self.hessian == 'reduced':
            i, j, Hx = ff.get_morse_potential_reduced_hessian(
                atoms, morse)
        elif self.hessian == 'spectral':
            i, j, Hx = ff.get_morse_potential_hessian(
                atoms, morse, spectral=True)
        else:
            raise NotImplementedError('Not implemented hessian')
        x = ij_to_x(i, j)
        row.extend(np.repeat(x, 6))
        col.extend(np.tile(x, 6))
        data.extend(Hx.flatten())
        if conn is not None:
            conn[i, j] = True
            conn[j, i] = True

    def add_bond(self, bond, atoms, row, col, data, conn=None):
        if self.hessian == 'reduced':
            i, j, Hx = ff.get_bond_potential_reduced_hessian(
                atoms, bond, self.morses)
        elif self.hessian == 'spectral':
            i, j, Hx = ff.get_bond_potential_hessian(
                atoms, bond, self.morses, spectral=True)
        else:
            raise NotImplementedError('Not implemented hessian')
        x = ij_to_x(i, j)
        row.extend(np.repeat(x, 6))
        col.extend(np.tile(x, 6))
        data.extend(Hx.flatten())
        if conn is not None:
            conn[i, j] = True
            conn[j, i] = True

    def add_angle(self, angle, atoms, row, col, data, conn=None):
        if self.hessian == 'reduced':
            i, j, k, Hx = ff.get_angle_potential_reduced_hessian(
                atoms, angle, self.morses)
        elif self.hessian == 'spectral':
            i, j, k, Hx = ff.get_angle_potential_hessian(
                atoms, angle, self.morses, spectral=True)
        else:
            raise NotImplementedError('Not implemented hessian')
        x = ijk_to_x(i, j, k)
        row.extend(np.repeat(x, 9))
        col.extend(np.tile(x, 9))
        data.extend(Hx.flatten())
        if conn is not None:
            conn[i, j] = conn[i, k] = conn[j, k] = True
            conn[j, i] = conn[k, i] = conn[k, j] = True

    def add_dihedral(self, dihedral, atoms, row, col, data, conn=None):
        if self.hessian == 'reduced':
            i, j, k, l, Hx = \
                ff.get_dihedral_potential_reduced_hessian(
                    atoms, dihedral, self.morses)
        elif self.hessian == 'spectral':
            i, j, k, l, Hx = ff.get_dihedral_potential_hessian(
                atoms, dihedral, self.morses, spectral=True)
        else:
            raise NotImplementedError('Not implemented hessian')
        x = ijkl_to_x(i, j, k, l)
        row.extend(np.repeat(x, 12))
        col.extend(np.tile(x, 12))
        data.extend(Hx.flatten())
        if conn is not None:
            conn[i, j] = conn[i, k] = conn[i, l] = conn[
                j, k] = conn[j, l] = conn[k, l] = True
            conn[j, i] = conn[k, i] = conn[l, i] = conn[
                k, j] = conn[l, j] = conn[l, k] = True

    def _make_sparse_precon(self, atoms, initial_assembly=False,
                            force_stab=False):
        N = len(atoms)

        row = []
        col = []
        data = []

        if self.morses is not None:
            for morse in self.morses:
                self.add_morse(morse, atoms, row, col, data)

        if self.bonds is not None:
            for bond in self.bonds:
                self.add_bond(bond, atoms, row, col, data)

        if self.angles is not None:
            for angle in self.angles:
                self.add_angle(angle, atoms, row, col, data)

        if self.dihedrals is not None:
            for dihedral in self.dihedrals:
                self.add_dihedral(dihedral, atoms, row, col, data)

        # add the diagonal
        row.extend(range(self.dim * N))
        col.extend(range(self.dim * N))
        data.extend([self.c_stab] * self.dim * N)

        # create the matrix
        start_time = time.time()
        self.P = sparse.csc_matrix(
            (data, (row, col)), shape=(self.dim * N, self.dim * N))
        self.logfile.write(
            f'--- created CSC matrix in {(time.time() - start_time)} s ---\n')

        self.P = apply_fixed(atoms, self.P)
        self.P = self.P.tocsr()
        self.logfile.write(
            f'--- N-dim precon created in {(time.time() - start_time)} s ---\n')
        self.create_solver()


class Exp_FF(Exp, FF):
    """Creates matrix with values decreasing exponentially with distance.
    """

    def __init__(self, A=3.0, r_cut=None, r_NN=None, mu=None, mu_c=None,
                 dim=3, c_stab=0.1,
                 force_stab=False, reinitialize=False, array_convention='C',
                 solver="auto", solve_tol=1e-9,
                 apply_positions=True, apply_cell=True,
                 estimate_mu_eigmode=False,
                 hessian='spectral', morses=None, bonds=None, angles=None,
                 dihedrals=None, logfile=None):
        """Initialise an Exp+FF preconditioner with given parameters.

        Args:
            r_cut, mu, c_stab, dim, reinitialize, array_convention: see
                precon.__init__()
            A: coefficient in exp(-A*r/r_NN). Default is A=3.0.
        """
        if (morses is None and bonds is None and angles is None and
                dihedrals is None):
            raise ImportError(
                'At least one of morses, bonds, angles or dihedrals must '
                'be defined!')

        SparsePrecon.__init__(self, r_cut=r_cut, r_NN=r_NN,
                              mu=mu, mu_c=mu_c, dim=dim, c_stab=c_stab,
                              force_stab=force_stab,
                              reinitialize=reinitialize,
                              array_convention=array_convention,
                              solver=solver,
                              solve_tol=solve_tol,
                              apply_positions=apply_positions,
                              apply_cell=apply_cell,
                              estimate_mu_eigmode=estimate_mu_eigmode,
                              logfile=logfile)

        self.A = A
        self.hessian = hessian
        self.morses = morses
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals

    def make_precon(self, atoms, reinitialize=None):
        if self.r_NN is None:
            self.r_NN = estimate_nearest_neighbour_distance(atoms,
                                                            self.neighbor_list)

        if self.r_cut is None:
            # This is the first time this function has been called, and no
            # cutoff radius has been specified, so calculate it automatically.
            self.r_cut = 2.0 * self.r_NN
        elif self.r_cut < self.r_NN:
            warning = ('WARNING: r_cut (%.2f) < r_NN (%.2f), '
                       'increasing to 1.1*r_NN = %.2f' % (self.r_cut,
                                                          self.r_NN,
                                                          1.1 * self.r_NN))
            warnings.warn(warning)
            self.r_cut = 1.1 * self.r_NN

        if reinitialize is None:
            # The caller has not specified whether or not to recalculate mu,
            # so the Precon's setting is used.
            reinitialize = self.reinitialize

        if self.mu is None:
            # Regardless of what the caller has specified, if we don't
            # currently have a value of mu, then we need one.
            reinitialize = True

        if reinitialize:
            self.estimate_mu(atoms)

        if self.P is not None:
            real_atoms = atoms
            if isinstance(atoms, Filter):
                real_atoms = atoms.atoms
            if self.old_positions is None:
                self.old_positions = real_atoms.positions
            displacement, _ = find_mic(real_atoms.positions -
                                       self.old_positions,
                                       real_atoms.cell, real_atoms.pbc)
            self.old_positions = real_atoms.get_positions()
            max_abs_displacement = abs(displacement).max()
            self.logfile.write('max(abs(displacements)) = %.2f A (%.2f r_NN)' %
                               (max_abs_displacement,
                                max_abs_displacement / self.r_NN))
            if max_abs_displacement < 0.5 * self.r_NN:
                return

        # Create the preconditioner:
        start_time = time.time()
        self._make_sparse_precon(atoms, force_stab=self.force_stab)
        self.logfile.write(
            f'--- Precon created in {(time.time() - start_time)} seconds ---\n')

    def _make_sparse_precon(self, atoms, initial_assembly=False,
                            force_stab=False):
        """Create a sparse preconditioner matrix based on the passed atoms.

        Args:
            atoms: the Atoms object used to create the preconditioner.

        Returns:
            A scipy.sparse.csr_matrix object, representing a d*N by d*N matrix
            (where N is the number of atoms, and d is the value of self.dim).
            BE AWARE that using numpy.dot() with this object will result in
            errors/incorrect results - use the .dot method directly on the
            sparse matrix instead.

        """
        self.logfile.write('creating sparse precon: initial_assembly=%r, '
                           'force_stab=%r, apply_positions=%r, '
                           'apply_cell=%r\n' %
                           (initial_assembly, force_stab,
                            self.apply_positions, self.apply_cell))

        N = len(atoms)
        start_time = time.time()
        if self.apply_positions:
            # compute neighbour list
            i_list, j_list, rij_list, _fixed_atoms = get_neighbours(
                atoms, self.r_cut, self.neighbor_list)
            self.logfile.write(
                f'--- neighbour list created in {(time.time() - start_time)} s '
                '---\n')

        row = []
        col = []
        data = []

        # precon is mu_c*identity for cell DoF
        start_time = time.time()
        if isinstance(atoms, Filter):
            i = N - 3
            j = N - 2
            k = N - 1
            x = ijk_to_x(i, j, k)
            row.extend(x)
            col.extend(x)
            if self.apply_cell:
                data.extend(np.repeat(self.mu_c, 9))
            else:
                data.extend(np.repeat(self.mu_c, 9))
        self.logfile.write(
            f'--- computed triplet format in {(time.time() - start_time)} s '
            '---\n')

        conn = sparse.lil_matrix((N, N), dtype=bool)

        if self.apply_positions and not initial_assembly:
            if self.morses is not None:
                for morse in self.morses:
                    self.add_morse(morse, atoms, row, col, data, conn)

            if self.bonds is not None:
                for bond in self.bonds:
                    self.add_bond(bond, atoms, row, col, data, conn)

            if self.angles is not None:
                for angle in self.angles:
                    self.add_angle(angle, atoms, row, col, data, conn)

            if self.dihedrals is not None:
                for dihedral in self.dihedrals:
                    self.add_dihedral(dihedral, atoms, row, col, data, conn)

        if self.apply_positions:
            for i, j, rij in zip(i_list, j_list, rij_list):
                if not conn[i, j]:
                    coeff = self.get_coeff(rij)
                    x = ij_to_x(i, j)
                    row.extend(x)
                    col.extend(x)
                    data.extend(3 * [-coeff] + 3 * [coeff])

        row.extend(range(self.dim * N))
        col.extend(range(self.dim * N))
        if initial_assembly:
            data.extend([self.mu * self.c_stab] * self.dim * N)
        else:
            data.extend([self.c_stab] * self.dim * N)

        # create the matrix
        start_time = time.time()
        self.P = sparse.csc_matrix(
            (data, (row, col)), shape=(self.dim * N, self.dim * N))
        self.logfile.write(
            f'--- created CSC matrix in {(time.time() - start_time)} s ---\n')

        if not initial_assembly:
            self.P = apply_fixed(atoms, self.P)

        self.P = self.P.tocsr()
        self.create_solver()


def make_precon(precon, atoms=None, **kwargs):
    """
    Construct preconditioner from a string and optionally build for Atoms

    Parameters
    ----------
    precon - one of 'C1', 'Exp', 'Pfrommer', 'FF', 'Exp_FF', 'ID', None
             or an instance of a subclass of `ase.optimize.precon.Precon`

    atoms - ase.atoms.Atoms instance, optional
            If present, build apreconditioner for this Atoms object

    **kwargs - additional keyword arguments to pass to Precon constructor

    Returns
    -------
    precon - instance of relevant subclass of `ase.optimize.precon.Precon`
    """
    lookup = {
        'C1': C1,
        'Exp': Exp,
        'Pfrommer': Pfrommer,
        'FF': FF,
        'Exp_FF': Exp_FF,
        'ID': IdentityPrecon,
        'IdentityPrecon': IdentityPrecon,
        None: IdentityPrecon
    }
    if isinstance(precon, str) or precon is None:
        cls = lookup[precon]
        precon = cls(**kwargs)
    if atoms is not None:
        precon.make_precon(atoms)
    return precon


class SplineFit:
    """
    Fit a cubic spline fit to images
    """

    def __init__(self, s, x):
        self._s = s
        self._x_data = x
        self._x = CubicSpline(self._s, x, bc_type='not-a-knot')
        self._dx_ds = self._x.derivative()
        self._d2x_ds2 = self._x.derivative(2)

    @property
    def s(self):
        return self._s

    @property
    def x_data(self):
        return self._x_data

    @property
    def x(self):
        return self._x

    @property
    def dx_ds(self):
        return self._dx_ds

    @property
    def d2x_ds2(self):
        return self._d2x_ds2


class PreconImages:
    def __init__(self, precon, images, **kwargs):
        """
        Wrapper for a list of Precon objects and associated images

        This is used when preconditioning a NEB object. Equation references
        refer to Paper IV in the :class:`ase.mep.NEB` documentation, i.e.

        S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
        150, 094109 (2019)
        https://dx.doi.org/10.1063/1.5064465

        Args:
            precon (str or list): preconditioner(s) to use
            images (list of Atoms): Atoms objects that define the state

        """
        self.images = images
        self._spline = None

        if isinstance(precon, list):
            if len(precon) != len(images):
                raise ValueError(f'length mismatch: len(precon)={len(precon)} '
                                 f'!= len(images)={len(images)}')
            self.precon = precon
            return
        P0 = make_precon(precon, images[0], **kwargs)
        self.precon = [P0]
        for image in images[1:]:
            P = P0.copy()
            P.make_precon(image)
            self.precon.append(P)

    def __len__(self):
        return len(self.precon)

    def __iter__(self):
        return iter(self.precon)

    def __getitem__(self, index):
        return self.precon[index]

    def apply(self, all_forces, index=None):
        """Apply preconditioners to stored images

        Args:
            all_forces (array): forces on images, shape (nimages, natoms, 3)
            index (slice, optional): Which images to include. Defaults to all.

        Returns:
            precon_forces: array of preconditioned forces
        """
        if index is None:
            index = slice(None)
        precon_forces = []
        for precon, image, forces in zip(self.precon[index],
                                         self.images[index],
                                         all_forces):
            f_vec = forces.reshape(-1)
            pf_vec, _ = precon.apply(f_vec, image)
            precon_forces.append(pf_vec.reshape(-1, 3))

        return np.array(precon_forces)

    def average_norm(self, i, j, dx):
        """Average norm between images i and j

        Args:
            i (int): left image
            j (int): right image
            dx (array): vector

        Returns:
            norm: norm of vector wrt average of precons at i and j
        """
        return np.sqrt(0.5 * (self.precon[i].dot(dx, dx) +
                              self.precon[j].dot(dx, dx)))

    def get_tangent(self, i):
        """Normalised tangent vector at image i

        Args:
            i (int): image of interest

        Returns:
            tangent: tangent vector, normalised with appropriate precon norm
        """
        tangent = self.spline.dx_ds(self.spline.s[i])
        tangent /= self.precon[i].norm(tangent)
        return tangent.reshape(-1, 3)

    def get_residual(self, i, imgforce):
        # residuals computed according to eq. 11 in the paper
        P_dot_imgforce = self.precon[i].Pdot(imgforce.reshape(-1))
        return np.linalg.norm(P_dot_imgforce, np.inf)

    def get_spring_force(self, i, k1, k2, tangent):
        """Spring force on image

        Args:
            i (int): image of interest
            k1 (float): spring constant for left spring
            k2 (float): spring constant for right spring
            tangent (array): tangent vector, shape (natoms, 3)

        Returns:
            eta: NEB spring forces, shape (natoms, 3)
        """
        # Definition following Eq. 9 in Paper IV
        nimages = len(self.images)
        k = 0.5 * (k1 + k2) / (nimages ** 2)
        curvature = self.spline.d2x_ds2(self.spline.s[i]).reshape(-1, 3)
        # complete Eq. 9 by including the spring force
        eta = k * self.precon[i].vdot(curvature, tangent) * tangent
        return eta

    def get_coordinates(self, positions=None):
        """Compute displacements wrt appropriate precon metric for each image

        Args:
            positions (list or array, optional) - images positions.
                Shape either (nimages * natoms, 3) or ((nimages-2)*natoms, 3)

        Returns:
            s : array shape (nimages,), reaction coordinates, in range [0, 1]
            x : array shape (nimages, 3 * natoms), flat displacement vectors
        """
        nimages = len(self.precon)
        natoms = len(self.images[0])
        d_P = np.zeros(nimages)
        x = np.zeros((nimages, 3 * natoms))  # flattened positions
        if positions is None:
            positions = [image.positions for image in self.images]
        elif isinstance(positions, np.ndarray) and len(positions.shape) == 2:
            positions = positions.reshape(-1, natoms, 3)
            positions = [positions[i, :, :] for i in range(len(positions))]
            if len(positions) == len(self.images) - 2:
                # prepend and append the non-moving images
                positions = ([self.images[0].positions] + positions +
                             [self.images[-1].positions])
        assert len(positions) == len(self.images)

        x[0, :] = positions[0].reshape(-1)
        for i in range(1, nimages):
            x[i, :] = positions[i].reshape(-1)
            dx, _ = find_mic(positions[i] - positions[i - 1],
                             self.images[i - 1].cell,
                             self.images[i - 1].pbc)
            dx = dx.reshape(-1)
            d_P[i] = self.average_norm(i, i - 1, dx)

        s = d_P.cumsum() / d_P.sum()  # Eq. A1 in paper IV
        return s, x

    def spline_fit(self, positions=None):
        """Fit 3 * natoms cubic splines as a function of reaction coordinate

        Returns:
            fit : :class:`ase.optimize.precon.SplineFit` object
        """
        s, x = self.get_coordinates(positions)
        return SplineFit(s, x)

    @property
    def spline(self):
        s, x = self.get_coordinates()
        if self._spline and (np.abs(s - self._old_s).max() < 1e-6 and
                             np.abs(x - self._old_x).max() < 1e-6):
            return self._spline

        self._spline = self.spline_fit()
        self._old_s = s
        self._old_x = x
        return self._spline
