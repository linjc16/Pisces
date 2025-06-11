# fmt: off

"""Module for calculating phonons of periodic systems."""

import warnings
from math import pi, sqrt
from pathlib import Path

import numpy as np
import numpy.fft as fft
import numpy.linalg as la

import ase
import ase.units as units
from ase.dft import monkhorst_pack
from ase.io.trajectory import Trajectory
from ase.parallel import world
from ase.utils import deprecated
from ase.utils.filecache import MultiFileJSONCache


class Displacement:
    """Abstract base class for phonon and el-ph supercell calculations.

    Both phonons and the electron-phonon interaction in periodic systems can be
    calculated with the so-called finite-displacement method where the
    derivatives of the total energy and effective potential are obtained from
    finite-difference approximations, i.e. by displacing the atoms. This class
    provides the required functionality for carrying out the calculations for
    the different displacements in its ``run`` member function.

    Derived classes must overwrite the ``__call__`` member function which is
    called for each atomic displacement.

    """

    def __init__(self, atoms, calc=None, supercell=(1, 1, 1), name=None,
                 delta=0.01, center_refcell=False, comm=None):
        """Init with an instance of class ``Atoms`` and a calculator.

        Parameters:

        atoms: Atoms object
            The atoms to work on.
        calc: Calculator
            Calculator for the supercell calculation.
        supercell: tuple
            Size of supercell given by the number of repetitions (l, m, n) of
            the small unit cell in each direction.
        name: str
            Base name to use for files.
        delta: float
            Magnitude of displacement in Ang.
        center_refcell: bool
            Reference cell in which the atoms will be displaced. If False, then
            corner cell in supercell is used. If True, then cell in the center
            of the supercell is used.
        comm: communicator
            MPI communicator for the phonon calculation.
            Default is to use world.
        """

        # Store atoms and calculator
        self.atoms = atoms
        self.calc = calc

        # Displace all atoms in the unit cell by default
        self.indices = np.arange(len(atoms))
        self.name = name
        self.delta = delta
        self.center_refcell = center_refcell
        self.supercell = supercell

        if comm is None:
            comm = world
        self.comm = comm

        self.cache = MultiFileJSONCache(self.name)

    def define_offset(self):        # Reference cell offset

        if not self.center_refcell:
            # Corner cell
            self.offset = 0
        else:
            # Center cell
            N_c = self.supercell
            self.offset = (N_c[0] // 2 * (N_c[1] * N_c[2]) +
                           N_c[1] // 2 * N_c[2] +
                           N_c[2] // 2)
        return self.offset

    @property
    @ase.utils.deprecated('Please use phonons.supercell instead of .N_c')
    def N_c(self):
        return self._supercell

    @property
    def supercell(self):
        return self._supercell

    @supercell.setter
    def supercell(self, supercell):
        assert len(supercell) == 3
        self._supercell = tuple(supercell)
        self.define_offset()
        self._lattice_vectors_array = self.compute_lattice_vectors()

    @ase.utils.deprecated('Please use phonons.compute_lattice_vectors()'
                          ' instead of .lattice_vectors()')
    def lattice_vectors(self):
        return self.compute_lattice_vectors()

    def compute_lattice_vectors(self):
        """Return lattice vectors for cells in the supercell."""
        # Lattice vectors -- ordered as illustrated in class docstring

        # Lattice vectors relevative to the reference cell
        R_cN = np.indices(self.supercell).reshape(3, -1)
        N_c = np.array(self.supercell)[:, np.newaxis]
        if self.offset == 0:
            R_cN += N_c // 2
            R_cN %= N_c
        R_cN -= N_c // 2
        return R_cN

    def __call__(self, *args, **kwargs):
        """Member function called in the ``run`` function."""

        raise NotImplementedError("Implement in derived classes!.")

    def set_atoms(self, atoms):
        """Set the atoms to vibrate.

        Parameters:

        atoms: list
            Can be either a list of strings, ints or ...

        """

        assert isinstance(atoms, list)
        assert len(atoms) <= len(self.atoms)

        if isinstance(atoms[0], str):
            assert np.all([isinstance(atom, str) for atom in atoms])
            sym_a = self.atoms.get_chemical_symbols()
            # List for atomic indices
            indices = []
            for type in atoms:
                indices.extend([a for a, atom in enumerate(sym_a)
                                if atom == type])
        else:
            assert np.all([isinstance(atom, int) for atom in atoms])
            indices = atoms

        self.indices = indices

    def _eq_disp(self):
        return self._disp(0, 0, 0)

    def _disp(self, a, i, step):
        from ase.vibrations.vibrations import Displacement as VDisplacement
        return VDisplacement(a, i, np.sign(step), abs(step), self)

    def run(self):
        """Run the calculations for the required displacements.

        This will do a calculation for 6 displacements per atom, +-x, +-y, and
        +-z. Only those calculations that are not already done will be
        started. Be aware that an interrupted calculation may produce an empty
        file (ending with .json), which must be deleted before restarting the
        job. Otherwise the calculation for that displacement will not be done.

        """

        # Atoms in the supercell -- repeated in the lattice vector directions
        # beginning with the last
        atoms_N = self.atoms * self.supercell

        # Set calculator if provided
        assert self.calc is not None, "Provide calculator in __init__ method"
        atoms_N.calc = self.calc

        # Do calculation on equilibrium structure
        eq_disp = self._eq_disp()
        with self.cache.lock(eq_disp.name) as handle:
            if handle is not None:
                output = self.calculate(atoms_N, eq_disp)
                handle.save(output)

        # Positions of atoms to be displaced in the reference cell
        natoms = len(self.atoms)
        offset = natoms * self.offset
        pos = atoms_N.positions[offset: offset + natoms].copy()

        # Loop over all displacements
        for a in self.indices:
            for i in range(3):
                for sign in [-1, 1]:
                    disp = self._disp(a, i, sign)
                    with self.cache.lock(disp.name) as handle:
                        if handle is None:
                            continue
                        try:
                            atoms_N.positions[offset + a, i] = \
                                pos[a, i] + sign * self.delta

                            result = self.calculate(atoms_N, disp)
                            handle.save(result)
                        finally:
                            # Return to initial positions
                            atoms_N.positions[offset + a, i] = pos[a, i]

        self.comm.barrier()

    def clean(self):
        """Delete generated files."""
        if self.comm.rank == 0:
            nfiles = self._clean()
        else:
            nfiles = 0
        self.comm.barrier()
        return nfiles

    def _clean(self):
        name = Path(self.name)

        nfiles = 0
        if name.is_dir():
            for fname in name.iterdir():
                fname.unlink()
                nfiles += 1
            name.rmdir()
        return nfiles


class Phonons(Displacement):
    r"""Class for calculating phonon modes using the finite displacement method.

    The matrix of force constants is calculated from the finite difference
    approximation to the first-order derivative of the atomic forces as::

                            2             nbj   nbj
                nbj        d E           F-  - F+
               C     = ------------ ~  -------------  ,
                mai     dR   dR          2 * delta
                          mai  nbj

    where F+/F- denotes the force in direction j on atom nb when atom ma is
    displaced in direction +i/-i. The force constants are related by various
    symmetry relations. From the definition of the force constants it must
    be symmetric in the three indices mai::

                nbj    mai         bj        ai
               C    = C      ->   C  (R ) = C  (-R )  .
                mai    nbj         ai  n     bj   n

    As the force constants can only depend on the difference between the m and
    n indices, this symmetry is more conveniently expressed as shown on the
    right hand-side.

    The acoustic sum-rule::

                           _ _
                aj         \    bj
               C  (R ) = -  )  C  (R )
                ai  0      /__  ai  m
                          (m, b)
                            !=
                          (0, a)

    Ordering of the unit cells illustrated here for a 1-dimensional system (in
    case ``refcell=None`` in constructor!):

    ::

               m = 0        m = 1        m = -2        m = -1
           -----------------------------------------------------
           |            |            |            |            |
           |        * b |        *   |        *   |        *   |
           |            |            |            |            |
           |   * a      |   *        |   *        |   *        |
           |            |            |            |            |
           -----------------------------------------------------

    Example:

    >>> from ase.build import bulk
    >>> from ase.phonons import Phonons
    >>> from gpaw import GPAW, FermiDirac

    >>> atoms = bulk('Si', 'diamond', a=5.4)
    >>> calc = GPAW(mode='fd',
    ...             kpts=(5, 5, 5),
    ...             h=0.2,
    ...             occupations=FermiDirac(0.))
    >>> ph = Phonons(atoms, calc, supercell=(5, 5, 5))
    >>> ph.run()
    >>> ph.read(method='frederiksen', acoustic=True)

    """

    def __init__(self, *args, **kwargs):
        """Initialize with base class args and kwargs."""

        if 'name' not in kwargs:
            kwargs['name'] = "phonon"

        self.deprecate_refcell(kwargs)

        Displacement.__init__(self, *args, **kwargs)

        # Attributes for force constants and dynamical matrix in real space
        self.C_N = None  # in units of eV / Ang**2
        self.D_N = None  # in units of eV / Ang**2 / amu

        # Attributes for born charges and static dielectric tensor
        self.Z_avv = None
        self.eps_vv = None

    @staticmethod
    def deprecate_refcell(kwargs: dict):
        if 'refcell' in kwargs:
            warnings.warn('Keyword refcell of Phonons is deprecated.'
                          'Please use center_refcell (bool)', FutureWarning)
            kwargs['center_refcell'] = bool(kwargs['refcell'])
            kwargs.pop('refcell')

        return kwargs

    def __call__(self, atoms_N):
        """Calculate forces on atoms in supercell."""
        return atoms_N.get_forces()

    def calculate(self, atoms_N, disp):
        forces = self(atoms_N)
        return {'forces': forces}

    def check_eq_forces(self):
        """Check maximum size of forces in the equilibrium structure."""

        eq_disp = self._eq_disp()
        feq_av = self.cache[eq_disp.name]['forces']

        fmin = feq_av.min()
        fmax = feq_av.max()
        i_min = np.where(feq_av == fmin)
        i_max = np.where(feq_av == fmax)

        return fmin, fmax, i_min, i_max

    @deprecated('Current implementation of non-analytical correction is '
                'likely incorrect, see '
                'https://gitlab.com/ase/ase/-/issues/941')
    def read_born_charges(self, name='born', neutrality=True):
        r"""Read Born charges and dieletric tensor from JSON file.

        The charge neutrality sum-rule::

                   _ _
                   \    a
                    )  Z   = 0
                   /__  ij
                    a

        Parameters:

        neutrality: bool
            Restore charge neutrality condition on calculated Born effective
            charges.
        name: str
            Key used to identify the file with Born charges for the unit cell
            in the JSON cache.

        .. deprecated:: 3.22.1
            Current implementation of non-analytical correction is likely
            incorrect, see :issue:`941`
        """

        # Load file with Born charges and dielectric tensor for atoms in the
        # unit cell
        Z_avv, eps_vv = self.cache[name]

        # Neutrality sum-rule
        if neutrality:
            Z_mean = Z_avv.sum(0) / len(Z_avv)
            Z_avv -= Z_mean

        self.Z_avv = Z_avv[self.indices]
        self.eps_vv = eps_vv

    def read(self, method='Frederiksen', symmetrize=3, acoustic=True,
             cutoff=None, born=False, **kwargs):
        """Read forces from json files and calculate force constants.

        Extra keyword arguments will be passed to ``read_born_charges``.

        Parameters:

        method: str
            Specify method for evaluating the atomic forces.
        symmetrize: int
            Symmetrize force constants (see doc string at top) when
            ``symmetrize != 0`` (default: 3). Since restoring the acoustic sum
            rule breaks the symmetry, the symmetrization must be repeated a few
            times until the changes a insignificant. The integer gives the
            number of iterations that will be carried out.
        acoustic: bool
            Restore the acoustic sum rule on the force constants.
        cutoff: None or float
            Zero elements in the dynamical matrix between atoms with an
            interatomic distance larger than the cutoff.
        born: bool
            Read in Born effective charge tensor and high-frequency static
            dielelctric tensor from file.

        """

        method = method.lower()
        assert method in ['standard', 'frederiksen']
        if cutoff is not None:
            cutoff = float(cutoff)

        # Read Born effective charges and optical dielectric tensor
        if born:
            self.read_born_charges(**kwargs)

        # Number of atoms
        natoms = len(self.indices)
        # Number of unit cells
        N = np.prod(self.supercell)
        # Matrix of force constants as a function of unit cell index in units
        # of eV / Ang**2
        C_xNav = np.empty((natoms * 3, N, natoms, 3), dtype=float)

        # Loop over all atomic displacements and calculate force constants
        for i, a in enumerate(self.indices):
            for j, v in enumerate('xyz'):
                # Atomic forces for a displacement of atom a in direction v
                # basename = '%s.%d%s' % (self.name, a, v)
                basename = '%d%s' % (a, v)
                fminus_av = self.cache[basename + '-']['forces']
                fplus_av = self.cache[basename + '+']['forces']

                if method == 'frederiksen':
                    fminus_av[a] -= fminus_av.sum(0)
                    fplus_av[a] -= fplus_av.sum(0)

                # Finite difference derivative
                C_av = fminus_av - fplus_av
                C_av /= 2 * self.delta

                # Slice out included atoms
                C_Nav = C_av.reshape((N, len(self.atoms), 3))[:, self.indices]
                index = 3 * i + j
                C_xNav[index] = C_Nav

        # Make unitcell index the first and reshape
        C_N = C_xNav.swapaxes(0, 1).reshape((N,) + (3 * natoms, 3 * natoms))

        # Cut off before symmetry and acoustic sum rule are imposed
        if cutoff is not None:
            self.apply_cutoff(C_N, cutoff)

        # Symmetrize force constants
        if symmetrize:
            for _ in range(symmetrize):
                # Symmetrize
                C_N = self.symmetrize(C_N)
                # Restore acoustic sum-rule
                if acoustic:
                    self.acoustic(C_N)
                else:
                    break

        # Store force constants and dynamical matrix
        self.C_N = C_N
        self.D_N = C_N.copy()

        # Add mass prefactor
        m_a = self.atoms.get_masses()
        self.m_inv_x = np.repeat(m_a[self.indices]**-0.5, 3)
        M_inv = np.outer(self.m_inv_x, self.m_inv_x)
        for D in self.D_N:
            D *= M_inv

    def symmetrize(self, C_N):
        """Symmetrize force constant matrix."""

        # Number of atoms
        natoms = len(self.indices)
        # Number of unit cells
        N = np.prod(self.supercell)

        # Reshape force constants to (l, m, n) cell indices
        C_lmn = C_N.reshape(self.supercell + (3 * natoms, 3 * natoms))

        # Shift reference cell to center index
        if self.offset == 0:
            C_lmn = fft.fftshift(C_lmn, axes=(0, 1, 2)).copy()
        # Make force constants symmetric in indices -- in case of an even
        # number of unit cells don't include the first cell
        i, j, k = 1 - np.asarray(self.supercell) % 2
        C_lmn[i:, j:, k:] *= 0.5
        C_lmn[i:, j:, k:] += \
            C_lmn[i:, j:, k:][::-1, ::-1, ::-1].transpose(0, 1, 2, 4, 3).copy()
        if self.offset == 0:
            C_lmn = fft.ifftshift(C_lmn, axes=(0, 1, 2)).copy()

        # Change to single unit cell index shape
        C_N = C_lmn.reshape((N, 3 * natoms, 3 * natoms))

        return C_N

    def acoustic(self, C_N):
        """Restore acoustic sumrule on force constants."""

        # Number of atoms
        natoms = len(self.indices)
        # Copy force constants
        C_N_temp = C_N.copy()

        # Correct atomic diagonals of R_m = (0, 0, 0) matrix
        for C in C_N_temp:
            for a in range(natoms):
                for a_ in range(natoms):
                    C_N[self.offset,
                        3 * a: 3 * a + 3,
                        3 * a: 3 * a + 3] -= C[3 * a: 3 * a + 3,
                                               3 * a_: 3 * a_ + 3]

    def apply_cutoff(self, D_N, r_c):
        """Zero elements for interatomic distances larger than the cutoff.

        Parameters:

        D_N: ndarray
            Dynamical/force constant matrix.
        r_c: float
            Cutoff in Angstrom.

        """

        # Number of atoms and primitive cells
        natoms = len(self.indices)
        N = np.prod(self.supercell)
        # Lattice vectors
        R_cN = self._lattice_vectors_array
        # Reshape matrix to individual atomic and cartesian dimensions
        D_Navav = D_N.reshape((N, natoms, 3, natoms, 3))

        # Cell vectors
        cell_vc = self.atoms.cell.transpose()
        # Atomic positions in reference cell
        pos_av = self.atoms.get_positions()

        # Zero elements with a distance to atoms in the reference cell
        # larger than the cutoff
        for n in range(N):
            # Lattice vector to cell
            R_v = np.dot(cell_vc, R_cN[:, n])
            # Atomic positions in cell
            posn_av = pos_av + R_v
            # Loop over atoms and zero elements
            for i, a in enumerate(self.indices):
                dist_a = np.sqrt(np.sum((pos_av[a] - posn_av)**2, axis=-1))
                # Atoms where the distance is larger than the cufoff
                i_a = dist_a > r_c  # np.where(dist_a > r_c)
                # Zero elements
                D_Navav[n, i, :, i_a, :] = 0.0

    def get_force_constant(self):
        """Return matrix of force constants."""

        assert self.C_N is not None
        return self.C_N

    def get_band_structure(self, path, modes=False, born=False, verbose=True):
        """Calculate and return the phonon band structure.

        This method computes the phonon band structure for a given path
        in reciprocal space. It is a wrapper around the internal
        `band_structure` method of the `Phonons` class. The method can
        optionally calculate and return phonon modes.

        Frequencies and modes are in units of eV and 1/sqrt(amu),
        respectively.

        Parameters:

        path : BandPath object
            The BandPath object defining the path in the reciprocal
            space over which the phonon band structure is calculated.
        modes : bool, optional
            If True, phonon modes will also be calculated and returned.
            Defaults to False.
        born : bool, optional
            If True, includes the effect of Born effective charges in
            the phonon calculations.
            Defaults to False.
        verbose : bool, optional
            If True, enables verbose output during the calculation.
            Defaults to True.

        Returns:

        BandStructure or tuple of (BandStructure, ndarray)
            If ``modes`` is False, returns a ``BandStructure`` object
            containing the phonon band structure. If ``modes`` is True,
            returns a tuple, where the first element is the
            ``BandStructure`` object and the second element is an ndarray
            of phonon modes.

            If modes are returned, the array is of shape
            (k-point, bands, atoms, 3) and the norm-squared of the mode
            is `1 / m_{eff}`, where `m_{eff}` is the effective mass of the
            mode.

        Example:

        >>> from ase.dft.kpoints import BandPath
        >>> path = BandPath(...)  # Define the band path
        >>> phonons = Phonons(...)
        >>> bs, modes = phonons.get_band_structure(path, modes=True)
        """
        result = self.band_structure(path.kpts,
                                     modes=modes,
                                     born=born,
                                     verbose=verbose)
        if modes:
            omega_kl, omega_modes = result
        else:
            omega_kl = result

        from ase.spectrum.band_structure import BandStructure
        bs = BandStructure(path, energies=omega_kl[None])

        # Return based on the modes flag
        return (bs, omega_modes) if modes else bs

    def compute_dynamical_matrix(self, q_scaled: np.ndarray, D_N: np.ndarray):
        """ Computation of the dynamical matrix in momentum space D_ab(q).
            This is a Fourier transform from real-space dynamical matrix D_N
            for a given momentum vector q.

        q_scaled: q vector in scaled coordinates.

        D_N: the dynamical matrix in real-space. It is necessary, at least
             currently, to provide this matrix explicitly (rather than use
             self.D_N) because this matrix is modified by the Born charges
             contributions and these modifications are momentum (q) dependent.

        Result:
            D(q): two-dimensional, complex-valued array of
                  shape=(3 * natoms, 3 * natoms).
        """
        # Evaluate fourier sum
        R_cN = self._lattice_vectors_array
        phase_N = np.exp(-2.j * pi * np.dot(q_scaled, R_cN))
        D_q = np.sum(phase_N[:, np.newaxis, np.newaxis] * D_N, axis=0)
        return D_q

    def band_structure(self, path_kc, modes=False, born=False, verbose=True):
        """Calculate phonon dispersion along a path in the Brillouin zone.

        The dynamical matrix at arbitrary q-vectors is obtained by Fourier
        transforming the real-space force constants. In case of negative
        eigenvalues (squared frequency), the corresponding negative frequency
        is returned.

        Frequencies and modes are in units of eV and 1/sqrt(amu),
        respectively.

        Parameters:

        path_kc: ndarray
            List of k-point coordinates (in units of the reciprocal lattice
            vectors) specifying the path in the Brillouin zone for which the
            dynamical matrix will be calculated.
        modes: bool
            Returns both frequencies and modes when True.
        born: bool
            Include non-analytic part given by the Born effective charges and
            the static part of the high-frequency dielectric tensor. This
            contribution to the force constant accounts for the splitting
            between the LO and TO branches for q -> 0.
        verbose: bool
            Print warnings when imaginary frequncies are detected.

        Returns:

        If modes=False: Array of energies

        If modes=True: Tuple of two arrays with energies and modes.
        """

        assert self.D_N is not None
        if born:
            assert self.Z_avv is not None
            assert self.eps_vv is not None

        # Dynamical matrix in real-space
        D_N = self.D_N

        # Lists for frequencies and modes along path
        omega_kl = []
        u_kl = []

        # Reciprocal basis vectors for use in non-analytic contribution
        reci_vc = 2 * pi * la.inv(self.atoms.cell)
        # Unit cell volume in Bohr^3
        vol = abs(la.det(self.atoms.cell)) / units.Bohr**3

        for q_c in path_kc:
            # Add non-analytic part
            if born:
                # q-vector in cartesian coordinates
                q_v = np.dot(reci_vc, q_c)
                # Non-analytic contribution to force constants in atomic units
                qdotZ_av = np.dot(q_v, self.Z_avv).ravel()
                C_na = (4 * pi * np.outer(qdotZ_av, qdotZ_av) /
                        np.dot(q_v, np.dot(self.eps_vv, q_v)) / vol)
                self.C_na = C_na / units.Bohr**2 * units.Hartree
                # Add mass prefactor and convert to eV / (Ang^2 * amu)
                M_inv = np.outer(self.m_inv_x, self.m_inv_x)
                D_na = C_na * M_inv / units.Bohr**2 * units.Hartree
                self.D_na = D_na
                D_N = self.D_N + D_na / np.prod(self.supercell)

            # if np.prod(self.N_c) == 1:
            #
            #     q_av = np.tile(q_v, len(self.indices))
            #     q_xx = np.vstack([q_av]*len(self.indices)*3)
            #     D_m += q_xx

            # Evaluate fourier sum
            D_q = self.compute_dynamical_matrix(q_c, D_N)

            if modes:
                omega2_l, u_xl = la.eigh(D_q, UPLO='U')
                # Sort eigenmodes according to eigenvalues (see below) and
                # multiply with mass prefactor.  This gives the eigenmode
                # (which is now not normalized!) with units 1/sqrt(amu).
                u_lx = (self.m_inv_x[:, np.newaxis] *
                        u_xl[:, omega2_l.argsort()]).T.copy()
                u_kl.append(u_lx.reshape((-1, len(self.indices), 3)))
            else:
                omega2_l = la.eigvalsh(D_q, UPLO='U')

            # Sort eigenvalues in increasing order
            omega2_l.sort()
            # Use dtype=complex to handle negative eigenvalues
            omega_l = np.sqrt(omega2_l.astype(complex))

            # Take care of imaginary frequencies
            if not np.all(omega2_l >= 0.):
                indices = np.where(omega2_l < 0)[0]

                if verbose:
                    print('WARNING, %i imaginary frequencies at '
                          'q = (% 5.2f, % 5.2f, % 5.2f) ; (omega_q =% 5.3e*i)'
                          % (len(indices), q_c[0], q_c[1], q_c[2],
                             omega_l[indices][0].imag))

                omega_l[indices] = -1 * np.sqrt(np.abs(omega2_l[indices].real))

            omega_kl.append(omega_l.real)

        # Conversion factor: sqrt(eV / Ang^2 / amu) -> eV
        s = units._hbar * 1e10 / sqrt(units._e * units._amu)
        omega_kl = s * np.asarray(omega_kl)

        if modes:
            return omega_kl, np.asarray(u_kl)

        return omega_kl

    def get_dos(self, kpts=(10, 10, 10), indices=None, verbose=True):
        """Return a phonon density of states.

        Parameters:

        kpts: tuple
            Shape of Monkhorst-Pack grid for sampling the Brillouin zone.
        indices: list
            If indices is not None, the amplitude-weighted atomic-partial
            DOS for the specified atoms will be calculated.
        verbose: bool
            Print warnings when imaginary frequncies are detected.

        Returns:
            A RawDOSData object containing the density of states.
        """
        from ase.spectrum.dosdata import RawDOSData

        # dos = self.dos(kpts, npts, delta, indices)
        kpts_kc = monkhorst_pack(kpts)
        if indices is None:
            # Return the total DOS
            omega_w = self.band_structure(kpts_kc, verbose=verbose)
            assert omega_w.ndim == 2
            n_kpt = omega_w.shape[0]
            omega_w = omega_w.ravel()
            dos = RawDOSData(omega_w, np.ones_like(omega_w) / n_kpt)
        else:
            # Return a partial DOS
            omegas, amplitudes = self.band_structure(kpts_kc,
                                                     modes=True,
                                                     verbose=verbose)
            # omegas.shape = (k-points, bands)
            # amplitudes.shape = (k-points, bands, atoms, 3)
            ampl_sq = (np.abs(amplitudes)**2).sum(axis=3)
            assert ampl_sq.ndim == 3
            assert ampl_sq.shape == omegas.shape + (len(self.indices),)
            weights = ampl_sq[:, :, indices].sum(axis=2) / ampl_sq.sum(axis=2)
            dos = RawDOSData(omegas.ravel(), weights.ravel() / omegas.shape[0])
        return dos

    @deprecated('Please use Phonons.get_dos() instead of Phonons.dos().')
    def dos(self, kpts=(10, 10, 10), npts=1000, delta=1e-3):
        """Calculate phonon dos as a function of energy.

        Parameters:

        kpts: tuple
            Shape of Monkhorst-Pack grid for sampling the Brillouin zone.
        npts: int
            Number of energy points.
        delta: float
            Broadening of Lorentzian line-shape in eV.

        Returns:
            Tuple of (frequencies, dos).  The frequencies are in units of eV.

        .. deprecated:: 3.23.1
            Please use the ``.get_dos()`` method instead, it returns a proper
            RawDOSData object.
        """

        # Monkhorst-Pack grid
        kpts_kc = monkhorst_pack(kpts)
        N = np.prod(kpts)
        # Get frequencies
        omega_kl = self.band_structure(kpts_kc)
        # Energy axis and dos
        omega_e = np.linspace(0., np.amax(omega_kl) + 5e-3, num=npts)
        dos_e = np.zeros_like(omega_e)

        # Sum up contribution from all q-points and branches
        for omega_l in omega_kl:
            diff_el = (omega_e[:, np.newaxis] - omega_l[np.newaxis, :])**2
            dos_el = 1. / (diff_el + (0.5 * delta)**2)
            dos_e += dos_el.sum(axis=1)

        dos_e *= 1. / (N * pi) * 0.5 * delta

        return omega_e, dos_e

    def write_modes(self, q_c, branches=0, kT=units.kB * 300, born=False,
                    repeat=(1, 1, 1), nimages=30, center=False):
        """Write modes to trajectory file.

        Parameters:

        q_c: ndarray of shape (3,)
            q-vector of the modes.
        branches: int or list
            Branch index of modes.
        kT: float
            Temperature in units of eV. Determines the amplitude of the atomic
            displacements in the modes.
        born: bool
            Include non-analytic contribution to the force constants at q -> 0.
        repeat: tuple
            Repeat atoms (l, m, n) times in the directions of the lattice
            vectors. Displacements of atoms in repeated cells carry a Bloch
            phase factor given by the q-vector and the cell lattice vector R_m.
        nimages: int
            Number of images in an oscillation.
        center: bool
            Center atoms in unit cell if True (default: False).

        To exaggerate the amplitudes for better visualization, multiply
        kT by the square of the desired factor.
        """

        if isinstance(branches, int):
            branch_l = [branches]
        else:
            branch_l = list(branches)

        # Calculate modes
        omega_l, u_l = self.band_structure([q_c], modes=True, born=born)
        # Repeat atoms
        atoms = self.atoms * repeat
        # Center
        if center:
            atoms.center()

        # Here ``Na`` refers to a composite unit cell/atom dimension
        pos_Nav = atoms.get_positions()
        # Total number of unit cells
        N = np.prod(repeat)

        # Corresponding lattice vectors R_m
        R_cN = np.indices(repeat).reshape(3, -1)
        # Bloch phase
        phase_N = np.exp(2.j * pi * np.dot(q_c, R_cN))
        phase_Na = phase_N.repeat(len(self.atoms))

        hbar = units._hbar * units.J * units.second
        for lval in branch_l:

            omega = omega_l[0, lval]
            u_av = u_l[0, lval]
            assert u_av.ndim == 2

            # For a classical harmonic oscillator, <x^2> = k T / m omega^2
            # and <x^2> = 1/2 u^2 where u is the amplitude and m is the
            # effective mass of the mode.
            # The reciprocal mass is already included in the normalization
            # of the modes.  The variable omega is actually hbar*omega (it
            # is in eV, not reciprocal ASE time units).
            u_av *= hbar * sqrt(2 * kT) / abs(omega)

            mode_av = np.zeros((len(self.atoms), 3), dtype=complex)
            # Insert slice with atomic displacements for the included atoms
            mode_av[self.indices] = u_av
            # Repeat and multiply by Bloch phase factor
            mode_Nav = np.vstack(N * [mode_av]) * phase_Na[:, np.newaxis]

            with Trajectory('%s.mode.%d.traj'
                            % (self.name, lval), 'w') as traj:
                for x in np.linspace(0, 2 * pi, nimages, endpoint=False):
                    atoms.set_positions((pos_Nav + np.exp(1.j * x) *
                                         mode_Nav).real)
                    traj.write(atoms)
