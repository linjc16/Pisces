# fmt: off

import numpy as np
from numpy.linalg import eigh, norm, pinv
from scipy.linalg import lstsq  # performs better than numpy.linalg.lstsq

from ase import units
from ase.calculators.calculator import (
    BaseCalculator,
    CalculationFailed,
    Calculator,
    CalculatorSetupError,
    all_changes,
)


class HarmonicCalculator(BaseCalculator):
    """Class for calculations with a Hessian-based harmonic force field.

    See :class:`HarmonicForceField` and the literature. [1]_
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, harmonicforcefield):
        """
        Parameters
        ----------
        harmonicforcefield: :class:`HarmonicForceField`
            Class for calculations with a Hessian-based harmonic force field.
        """
        super().__init__()  # parameters have been passed to the force field
        self.harmonicforcefield = harmonicforcefield

    def calculate(self, atoms, properties, system_changes):
        self.atoms = atoms.copy()  # for caching of results
        energy, forces_x = self.harmonicforcefield.get_energy_forces(atoms)
        self.results['energy'] = energy
        self.results['forces'] = forces_x


class HarmonicForceField:
    def __init__(self, ref_atoms, hessian_x, ref_energy=0.0, get_q_from_x=None,
                 get_jacobian=None, cartesian=True, variable_orientation=False,
                 hessian_limit=0.0, constrained_q=None, rcond=1e-7,
                 zero_thresh=0.0):
        """Class that represents a Hessian-based harmonic force field.

        Energy and forces of this force field are based on the
        Cartesian Hessian for a local reference configuration, i.e. if
        desired, on the Hessian matrix transformed to a user-defined
        coordinate system.  The required Hessian has to be passed as
        an argument, e.g. predetermined numerically via central finite
        differences in Cartesian coordinates.  Note that a potential
        being harmonic in Cartesian coordinates **x** is not
        necessarily equivalently harmonic in another coordinate system
        **q**, e.g. when the transformation between the coordinate
        systems is non-linear.  By default, the force field is
        evaluated in Cartesian coordinates in which energy and forces
        are not rotationally and translationally invariant.  Systems
        with variable orientation, require rotationally and
        translationally invariant calculations for which a set of
        appropriate coordinates has to be defined. This can be a set
        of (redundant) internal coordinates (bonds, angles, dihedrals,
        coordination numbers, ...) or any other user-defined
        coordinate system.

        Together with the :class:`HarmonicCalculator` this
        :class:`HarmonicForceField` can be used to compute
        Anharmonic Corrections to the Harmonic Approximation. [1]_

        Parameters
        ----------
        ref_atoms: :class:`~ase.Atoms` object
            Reference structure for which energy (``ref_energy``) and Hessian
            matrix in Cartesian coordinates (``hessian_x``) are provided.

        hessian_x: numpy array
            Cartesian Hessian matrix for the reference structure ``ref_atoms``.
            If a user-defined coordinate system is provided via
            ``get_q_from_x`` and ``get_jacobian``, the Cartesian Hessian matrix
            is transformed to the user-defined coordinate system and back to
            Cartesian coordinates, thereby eliminating rotational and
            translational traits from the Hessian. The Hessian matrix
            obtained after this double-transformation is then used as
            the reference Hessian matrix to evaluate energy and forces for
            ``cartesian = True``. For ``cartesian = False`` the reference
            Hessian matrix transformed to the user-defined coordinates is used
            to compute energy and forces.

        ref_energy: float
            Energy of the reference structure ``ref_atoms``, typically in `eV`.

        get_q_from_x: python function, default: None (Cartesian coordinates)
            Function that returns a vector of user-defined coordinates **q** for
            a given :class:`~ase.Atoms` object 'atoms'. The signature should be:
            :obj:`get_q_from_x(atoms)`.

        get_jacobian: python function, default: None (Cartesian coordinates)
            Function that returns the geometric Jacobian matrix of the
            user-defined coordinates **q** w.r.t. Cartesian coordinates **x**
            defined as `dq/dx` (Wilson B-matrix) for a given :class:`~ase.Atoms`
            object 'atoms'. The signature should be: :obj:`get_jacobian(atoms)`.

        cartesian: bool
            Set to True to evaluate energy and forces based on the reference
            Hessian (system harmonic in Cartesian coordinates).
            Set to False to evaluate energy and forces based on the reference
            Hessian transformed to user-defined coordinates (system harmonic in
            user-defined coordinates).

        hessian_limit: float
            Reconstruct the reference Hessian matrix with a lower limit for the
            eigenvalues, typically in `eV/A^2`. Eigenvalues in the interval
            [``zero_thresh``, ``hessian_limit``] are set to ``hessian_limit``
            while the eigenvectors are left untouched.

        variable_orientation: bool
            Set to True if the investigated :class:`~ase.Atoms` has got
            rotational degrees of freedom such that the orientation with respect
            to ``ref_atoms`` might be different (typically for molecules).
            Set to False to speed up the calculation when ``cartesian = True``.

        constrained_q: list
            A list of indices 'i' of constrained coordinates `q_i` to be
            projected out from the Hessian matrix
            (e.g. remove forces along imaginary mode of a transition state).

        rcond: float
            Cutoff for singular value decomposition in the computation of the
            Moore-Penrose pseudo-inverse during transformation of the Hessian
            matrix. Equivalent to the rcond parameter in scipy.linalg.lstsq.

        zero_thresh: float
            Reconstruct the reference Hessian matrix with absolute eigenvalues
            below this threshold set to zero.

        """
        self.check_input([get_q_from_x, get_jacobian],
                         variable_orientation, cartesian)

        self.parameters = {'ref_atoms': ref_atoms,
                           'ref_energy': ref_energy,
                           'hessian_x': hessian_x,
                           'hessian_limit': hessian_limit,
                           'get_q_from_x': get_q_from_x,
                           'get_jacobian': get_jacobian,
                           'cartesian': cartesian,
                           'variable_orientation': variable_orientation,
                           'constrained_q': constrained_q,
                           'rcond': rcond,
                           'zero_thresh': zero_thresh}

        # set up user-defined coordinate system or Cartesian coordinates
        self.get_q_from_x = (self.parameters['get_q_from_x'] or
                             (lambda atoms: atoms.get_positions()))
        self.get_jacobian = (self.parameters['get_jacobian'] or
                             (lambda atoms: np.diagflat(
                                 np.ones(3 * len(atoms)))))

        # reference Cartesian coords. x0; reference user-defined coords. q0
        self.x0 = self.parameters['ref_atoms'].get_positions().ravel()
        self.q0 = self.get_q_from_x(self.parameters['ref_atoms']).ravel()
        self.setup_reference_hessians()  # self._hessian_x and self._hessian_q

        # store number of zero eigenvalues of G-matrix for redundancy check
        jac0 = self.get_jacobian(self.parameters['ref_atoms'])
        Gmat = jac0.T @ jac0
        self.Gmat_eigvals, _ = eigh(Gmat)  # stored for inspection purposes
        self.zero_eigvals = len(np.flatnonzero(np.abs(self.Gmat_eigvals) <
                                               self.parameters['zero_thresh']))

    @staticmethod
    def check_input(coord_functions, variable_orientation, cartesian):
        if None in coord_functions:
            if not all(func is None for func in coord_functions):
                msg = ('A user-defined coordinate system requires both '
                       '`get_q_from_x` and `get_jacobian`.')
                raise CalculatorSetupError(msg)
            if variable_orientation:
                msg = ('The use of `variable_orientation` requires a '
                       'user-defined, translationally and rotationally '
                       'invariant coordinate system.')
                raise CalculatorSetupError(msg)
            if not cartesian:
                msg = ('A user-defined coordinate system is required for '
                       'calculations with cartesian=False.')
                raise CalculatorSetupError(msg)

    def setup_reference_hessians(self):
        """Prepare projector to project out constrained user-defined coordinates
        **q** from Hessian. Then do transformation to user-defined coordinates
        and back. Relevant literature:
        * Peng, C. et al. J. Comput. Chem. 1996, 17 (1), 49-56.
        * Baker, J. et al. J. Chem. Phys. 1996, 105 (1), 192–212."""
        jac0 = self.get_jacobian(
            self.parameters['ref_atoms'])  # Jacobian (dq/dx)
        jac0 = self.constrain_jac(jac0)  # for reference Cartesian coordinates
        ijac0 = self.get_ijac(jac0, self.parameters['rcond'])
        self.transform2reference_hessians(jac0, ijac0)  # perform projection

    def constrain_jac(self, jac):
        """Procedure by Peng, Ayala, Schlegel and Frisch adjusted for redundant
        coordinates.
        Peng, C. et al. J. Comput. Chem. 1996, 17 (1), 49–56.
        """
        proj = jac @ jac.T  # build non-redundant projector
        constrained_q = self.parameters['constrained_q'] or []
        Cmat = np.zeros(proj.shape)  # build projector for constraints
        Cmat[constrained_q, constrained_q] = 1.0
        proj = proj - proj @ Cmat @ pinv(Cmat @ proj @ Cmat) @ Cmat @ proj
        jac = pinv(jac) @ proj  # come back to redundant projector
        return jac.T

    def transform2reference_hessians(self, jac0, ijac0):
        """Transform Cartesian Hessian matrix to user-defined coordinates
        and back to Cartesian coordinates. For suitable coordinate systems
        (e.g. internals) this removes rotational and translational degrees of
        freedom. Furthermore, apply the lower limit to the force constants
        and reconstruct Hessian matrix."""
        hessian_x = self.parameters['hessian_x']
        hessian_x = 0.5 * (hessian_x + hessian_x.T)  # guarantee symmetry
        hessian_q = ijac0.T @ hessian_x @ ijac0  # forward transformation
        hessian_x = jac0.T @ hessian_q @ jac0  # backward transformation
        hessian_x = 0.5 * (hessian_x + hessian_x.T)  # guarantee symmetry
        w, v = eigh(hessian_x)  # rot. and trans. degrees of freedom are removed
        w[np.abs(w) < self.parameters['zero_thresh']] = 0.0  # noise-cancelling
        w[(w > 0.0) & (w < self.parameters['hessian_limit'])] = self.parameters[
            'hessian_limit'
        ]
        # reconstruct Hessian from new eigenvalues and preserved eigenvectors
        hessian_x = v @ np.diagflat(w) @ v.T  # v.T == inv(v) due to symmetry
        self._hessian_x = 0.5 * (hessian_x + hessian_x.T)  # guarantee symmetry
        self._hessian_q = ijac0.T @ self._hessian_x @ ijac0

    @staticmethod
    def get_ijac(jac, rcond):  # jac is the Wilson B-matrix
        """Compute Moore-Penrose pseudo-inverse of Wilson B-matrix."""
        jac_T = jac.T  # btw. direct Jacobian inversion is slow, hence form Gmat
        Gmat = jac_T @ jac   # avoid: numpy.linalg.pinv(Gmat, rcond) @ jac_T
        ijac = lstsq(Gmat, jac_T, rcond, lapack_driver='gelsy')
        return ijac[0]  # [-1] would be eigenvalues of Gmat

    def get_energy_forces(self, atoms):
        """Return a tuple with energy and forces in Cartesian coordinates for
        a given :class:`~ase.Atoms` object."""
        q = self.get_q_from_x(atoms).ravel()

        if self.parameters['cartesian']:
            x = atoms.get_positions().ravel()
            x0 = self.x0
            hessian_x = self._hessian_x

            if self.parameters['variable_orientation']:
                # determine x0 for present orientation
                x0 = self.back_transform(x, q, self.q0, atoms.copy())
                ref_atoms = atoms.copy()
                ref_atoms.set_positions(x0.reshape(int(len(x0) / 3), 3))
                x0 = ref_atoms.get_positions().ravel()
                # determine jac0 for present orientation
                jac0 = self.get_jacobian(ref_atoms)
                self.check_redundancy(jac0)  # check for coordinate failure
                # determine hessian_x for present orientation
                hessian_x = jac0.T @ self._hessian_q @ jac0

            xdiff = x - x0
            forces_x = -hessian_x @ xdiff
            energy = -0.5 * (forces_x * xdiff).sum()

        else:
            jac = self.get_jacobian(atoms)
            self.check_redundancy(jac)  # check for coordinate failure
            qdiff = q - self.q0
            forces_q = -self._hessian_q @ qdiff
            forces_x = forces_q @ jac
            energy = -0.5 * (forces_q * qdiff).sum()

        energy += self.parameters['ref_energy']
        forces_x = forces_x.reshape(int(forces_x.size / 3), 3)
        return energy, forces_x

    def back_transform(self, x, q, q0, atoms_copy):
        """Find the right orientation in Cartesian reference coordinates."""
        xk = 1 * x
        qk = 1 * q
        dq = qk - q0
        err = abs(dq).max()
        count = 0
        atoms_copy.set_constraint()  # helpful for back-transformation
        while err > 1e-7:  # back-transformation tolerance for convergence
            count += 1
            if count > 99:  # maximum number of iterations during back-transf.
                msg = ('Back-transformation from user-defined to Cartesian '
                       'coordinates failed.')
                raise CalculationFailed(msg)
            jac = self.get_jacobian(atoms_copy)
            ijac = self.get_ijac(jac, self.parameters['rcond'])
            dx = ijac @ dq
            xk = xk - dx
            atoms_copy.set_positions(xk.reshape(int(len(xk) / 3), 3))
            qk = self.get_q_from_x(atoms_copy).ravel()
            dq = qk - q0
            err = abs(dq).max()
        return xk

    def check_redundancy(self, jac):
        """Compare number of zero eigenvalues of G-matrix to initial number."""
        Gmat = jac.T @ jac
        self.Gmat_eigvals, _ = eigh(Gmat)
        zero_eigvals = len(np.flatnonzero(np.abs(self.Gmat_eigvals) <
                                          self.parameters['zero_thresh']))
        if zero_eigvals != self.zero_eigvals:
            raise CalculationFailed('Suspected coordinate failure: '
                                    f'G-matrix has got {zero_eigvals} '
                                    'zero eigenvalues, but had '
                                    f'{self.zero_eigvals} during setup')

    @property
    def hessian_x(self):
        return self._hessian_x

    @property
    def hessian_q(self):
        return self._hessian_q


class SpringCalculator(Calculator):
    """
    Spring calculator corresponding to independent oscillators with a fixed
    spring constant.


    Energy for an atom is given as

    E = k / 2 * (r - r_0)**2

    where k is the spring constant and, r_0 the ideal positions.


    Parameters
    ----------
    ideal_positions : array
        array of the ideal crystal positions
    k : float
        spring constant in eV/Angstrom
    """
    implemented_properties = ['forces', 'energy', 'free_energy']

    def __init__(self, ideal_positions, k):
        Calculator.__init__(self)
        self.ideal_positions = ideal_positions.copy()
        self.k = k

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energy, forces = self.compute_energy_and_forces(atoms)
        self.results['energy'], self.results['forces'] = energy, forces

    def compute_energy_and_forces(self, atoms):
        disps = atoms.positions - self.ideal_positions
        forces = - self.k * disps
        energy = sum(self.k / 2.0 * norm(disps, axis=1)**2)
        return energy, forces

    def get_free_energy(self, T, method='classical'):
        """Get analytic vibrational free energy for the spring system.

        Parameters
        ----------
        T : float
            temperature (K)
        method : str
            method for free energy computation; 'classical' or 'QM'.
        """
        F = 0.0
        masses, counts = np.unique(self.atoms.get_masses(), return_counts=True)
        for m, c in zip(masses, counts):
            F += c * \
                SpringCalculator.compute_Einstein_solid_free_energy(
                    self.k, m, T, method)
        return F

    @staticmethod
    def compute_Einstein_solid_free_energy(k, m, T, method='classical'):
        """ Get free energy (per atom) for an Einstein crystal.

        Free energy of a Einstein solid given by classical (1) or QM (2)
        1.    F_E = 3NkbT log( hw/kbT )
        2.    F_E = 3NkbT log( 1-exp(hw/kbT) ) + zeropoint

        Parameters
        -----------
        k : float
            spring constant (eV/A^2)
        m : float
            mass (grams/mole or AMU)
        T : float
            temperature (K)
        method : str
            method for free energy computation, classical or QM.

        Returns
        --------
        float
            free energy of the Einstein crystal (eV/atom)
        """
        assert method in ['classical', 'QM']

        hbar = units._hbar * units.J  # eV/s
        m = m / units.kg              # mass kg
        k = k * units.m**2 / units.J  # spring constant J/m2
        omega = np.sqrt(k / m)        # angular frequency 1/s

        if method == 'classical':
            F_einstein = 3 * units.kB * T * \
                np.log(hbar * omega / (units.kB * T))
        elif method == 'QM':
            log_factor = np.log(1.0 - np.exp(-hbar * omega / (units.kB * T)))
            F_einstein = 3 * units.kB * T * log_factor + 1.5 * hbar * omega

        return F_einstein
