# fmt: off

import numpy as np

import ase.units as u
from ase.dft import monkhorst_pack
from ase.parallel import world
from ase.phonons import Phonons
from ase.utils import IOContext
from ase.vibrations.vibrations import AtomicDisplacements, Vibrations


class RamanCalculatorBase(IOContext):
    def __init__(self, atoms,  # XXX do we need atoms at this stage ?
                 *args,
                 name='raman',
                 exext='.alpha',
                 txt='-',
                 verbose=False,
                 comm=world,
                 **kwargs):
        """
        Parameters
        ----------
        atoms: ase Atoms object
        exext: string
          Extension for excitation filenames
        txt:
          Output stream
        verbose:
          Verbosity level of output
        comm:
          Communicator, default world
        """
        kwargs['name'] = name
        self.exname = kwargs.pop('exname', name)

        super().__init__(atoms, *args, **kwargs)

        self.exext = exext

        self.txt = self.openfile(file=txt, comm=comm)
        self.verbose = verbose

        self.comm = comm


class StaticRamanCalculatorBase(RamanCalculatorBase):
    """Base class for Raman intensities derived from
    static polarizabilities"""

    def __init__(self, atoms, exobj, exkwargs=None, *args, **kwargs):
        self.exobj = exobj
        if exkwargs is None:
            exkwargs = {}
        self.exkwargs = exkwargs
        super().__init__(atoms, *args, **kwargs)

    def _new_exobj(self):
        return self.exobj(**self.exkwargs)

    def calculate(self, atoms, disp):
        returnvalue = super().calculate(atoms, disp)
        disp.calculate_and_save_static_polarizability(atoms)
        return returnvalue


class StaticRamanCalculator(StaticRamanCalculatorBase, Vibrations):
    pass


class StaticRamanPhononsCalculator(StaticRamanCalculatorBase, Phonons):
    pass


class RamanBase(AtomicDisplacements, IOContext):
    def __init__(self, atoms,  # XXX do we need atoms at this stage ?
                 *args,
                 name='raman',
                 exname=None,
                 exext='.alpha',
                 txt='-',
                 verbose=False,
                 comm=world,
                 **kwargs):
        """
        Parameters
        ----------
        atoms: ase Atoms object
        exext: string
          Extension for excitation filenames
        txt:
          Output stream
        verbose:
          Verbosity level of output
        comm:
          Communicator, default world
        """
        self.atoms = atoms

        self.name = name
        if exname is None:
            self.exname = name
        else:
            self.exname = exname
        self.exext = exext

        self.txt = self.openfile(file=txt, comm=comm)
        self.verbose = verbose

        self.comm = comm


class RamanData(RamanBase):
    """Base class to evaluate Raman spectra from pre-computed data"""

    def __init__(self, atoms,  # XXX do we need atoms at this stage ?
                 *args,
                 exname=None,      # name for excited state calculations
                 **kwargs):
        """
        Parameters
        ----------
        atoms: ase Atoms object
        exname: string
            name for excited state calculations (defaults to name),
            used for reading excitations
        """
        super().__init__(atoms, *args, **kwargs)

        if exname is None:
            exname = kwargs.get('name', self.name)
        self.exname = exname
        self._already_read = False

    def get_energies(self):
        self.calculate_energies_and_modes()
        return self.om_Q

    def init_parallel_read(self):
        """Initialize variables for parallel read"""
        rank = self.comm.rank
        indices = self.indices
        myn = -(-self.ndof // self.comm.size)  # ceil divide
        self.slize = s = slice(myn * rank, myn * (rank + 1))
        self.myindices = np.repeat(indices, 3)[s]
        self.myxyz = ('xyz' * len(indices))[s]
        self.myr = range(self.ndof)[s]
        self.mynd = len(self.myr)

    def read(self, *args, **kwargs):
        """Read data from a pre-performed calculation."""
        if self._already_read:
            return

        self.vibrations.read(*args, **kwargs)
        self.init_parallel_read()
        self.read_excitations()

        self._already_read = True

    @staticmethod
    def m2(z):
        return (z * z.conj()).real

    def map_to_modes(self, V_rcc):
        V_qcc = (V_rcc.T * self.im_r).T  # units Angstrom^2 / sqrt(amu)
        V_Qcc = np.dot(V_qcc.T, self.modes_Qq.T).T
        return V_Qcc

    def me_Qcc(self, *args, **kwargs):
        """Full matrix element

        Returns
        -------
        Matrix element in e^2 Angstrom^2 / eV
        """
        # Angstrom^2 / sqrt(amu)
        elme_Qcc = self.electronic_me_Qcc(*args, **kwargs)
        # Angstrom^3 -> e^2 Angstrom^2 / eV
        elme_Qcc /= u.Hartree * u.Bohr  # e^2 Angstrom / eV / sqrt(amu)
        return elme_Qcc * self.vib01_Q[:, None, None]

    def get_absolute_intensities(self, delta=0, **kwargs):
        """Absolute Raman intensity or Raman scattering factor

        Parameter
        ---------
        delta: float
           pre-factor for asymmetric anisotropy, default 0

        References
        ----------
        Porezag and Pederson, PRB 54 (1996) 7830-7836 (delta=0)
        Baiardi and Barone, JCTC 11 (2015) 3267-3280 (delta=5)

        Returns
        -------
        raman intensity, unit Ang**4/amu
        """
        alpha2_r, gamma2_r, delta2_r = self._invariants(
            self.electronic_me_Qcc(**kwargs))
        return 45 * alpha2_r + delta * delta2_r + 7 * gamma2_r

    def intensity(self, *args, **kwargs):
        """Raman intensity

        Returns
        -------
        unit e^4 Angstrom^4 / eV^2
        """
        self.calculate_energies_and_modes()

        m2 = Raman.m2
        alpha_Qcc = self.me_Qcc(*args, **kwargs)
        if not self.observation:  # XXXX remove
            """Simple sum, maybe too simple"""
            return m2(alpha_Qcc).sum(axis=1).sum(axis=1)
        # XXX enable when appropriate
        #        if self.observation['orientation'].lower() != 'random':
        #            raise NotImplementedError('not yet')

        # random orientation of the molecular frame
        # Woodward & Long,
        # Guthmuller, J. J. Chem. Phys. 2016, 144 (6), 64106
        alpha2_r, gamma2_r, delta2_r = self._invariants(alpha_Qcc)

        if self.observation['geometry'] == '-Z(XX)Z':  # Porto's notation
            return (45 * alpha2_r + 5 * delta2_r + 4 * gamma2_r) / 45.
        elif self.observation['geometry'] == '-Z(XY)Z':  # Porto's notation
            return gamma2_r / 15.
        elif self.observation['scattered'] == 'Z':
            # scattered light in direction of incoming light
            return (45 * alpha2_r + 5 * delta2_r + 7 * gamma2_r) / 45.
        elif self.observation['scattered'] == 'parallel':
            # scattered light perendicular and
            # polarization in plane
            return 6 * gamma2_r / 45.
        elif self.observation['scattered'] == 'perpendicular':
            # scattered light perendicular and
            # polarization out of plane
            return (45 * alpha2_r + 5 * delta2_r + 7 * gamma2_r) / 45.
        else:
            raise NotImplementedError

    def _invariants(self, alpha_Qcc):
        """Raman invariants

        Parameter
        ---------
        alpha_Qcc: array
           Matrix element or polarizability tensor

        Reference
        ---------
        Derek A. Long, The Raman Effect, ISBN 0-471-49028-8

        Returns
        -------
        mean polarizability, anisotropy, asymmetric anisotropy
        """
        m2 = Raman.m2
        alpha2_r = m2(alpha_Qcc[:, 0, 0] + alpha_Qcc[:, 1, 1] +
                      alpha_Qcc[:, 2, 2]) / 9.
        delta2_r = 3 / 4. * (
            m2(alpha_Qcc[:, 0, 1] - alpha_Qcc[:, 1, 0]) +
            m2(alpha_Qcc[:, 0, 2] - alpha_Qcc[:, 2, 0]) +
            m2(alpha_Qcc[:, 1, 2] - alpha_Qcc[:, 2, 1]))
        gamma2_r = (3 / 4. * (m2(alpha_Qcc[:, 0, 1] + alpha_Qcc[:, 1, 0]) +
                              m2(alpha_Qcc[:, 0, 2] + alpha_Qcc[:, 2, 0]) +
                              m2(alpha_Qcc[:, 1, 2] + alpha_Qcc[:, 2, 1])) +
                    (m2(alpha_Qcc[:, 0, 0] - alpha_Qcc[:, 1, 1]) +
                     m2(alpha_Qcc[:, 0, 0] - alpha_Qcc[:, 2, 2]) +
                     m2(alpha_Qcc[:, 1, 1] - alpha_Qcc[:, 2, 2])) / 2)
        return alpha2_r, gamma2_r, delta2_r

    def summary(self, log='-'):
        """Print summary for given omega [eV]"""
        with IOContext() as io:
            log = io.openfile(file=log, mode='a', comm=self.comm)
            return self._summary(log)

    def _summary(self, log):
        hnu = self.get_energies()
        intensities = self.get_absolute_intensities()
        te = int(np.log10(intensities.max())) - 2
        scale = 10**(-te)

        if not te:
            ts = ''
        elif te > -2 and te < 3:
            ts = str(10**te)
        else:
            ts = f'10^{te}'

        print('-------------------------------------', file=log)
        print(' Mode    Frequency        Intensity', file=log)
        print(f'  #    meV     cm^-1      [{ts}A^4/amu]', file=log)
        print('-------------------------------------', file=log)
        for n, e in enumerate(hnu):
            if e.imag != 0:
                c = 'i'
                e = e.imag
            else:
                c = ' '
                e = e.real
            print('%3d %6.1f%s  %7.1f%s  %9.2f' %
                  (n, 1000 * e, c, e / u.invcm, c, intensities[n] * scale),
                  file=log)
        print('-------------------------------------', file=log)
        # XXX enable this in phonons
        # parprint('Zero-point energy: %.3f eV' %
        #         self.vibrations.get_zero_point_energy(),
        #         file=log)


class Raman(RamanData):
    def __init__(self, atoms, *args, **kwargs):
        super().__init__(atoms, *args, **kwargs)

        for key in ['txt', 'exext', 'exname']:
            kwargs.pop(key, None)
        kwargs['name'] = kwargs.get('name', self.name)
        self.vibrations = Vibrations(atoms, *args, **kwargs)

        self.delta = self.vibrations.delta
        self.indices = self.vibrations.indices

    def calculate_energies_and_modes(self):
        if hasattr(self, 'im_r'):
            return

        self.read()

        self.im_r = self.vibrations.im
        self.modes_Qq = self.vibrations.modes
        self.om_Q = self.vibrations.hnu.real    # energies in eV
        self.H = self.vibrations.H   # XXX used in albrecht.py
        # pre-factors for one vibrational excitation
        with np.errstate(divide='ignore'):
            self.vib01_Q = np.where(self.om_Q > 0,
                                    1. / np.sqrt(2 * self.om_Q), 0)
        # -> sqrt(amu) * Angstrom
        self.vib01_Q *= np.sqrt(u.Ha * u._me / u._amu) * u.Bohr


class RamanPhonons(RamanData):
    def __init__(self, atoms, *args, **kwargs):
        RamanData.__init__(self, atoms, *args, **kwargs)

        for key in ['txt', 'exext', 'exname']:
            kwargs.pop(key, None)
        kwargs['name'] = kwargs.get('name', self.name)
        self.vibrations = Phonons(atoms, *args, **kwargs)

        self.delta = self.vibrations.delta
        self.indices = self.vibrations.indices

        self.kpts = (1, 1, 1)

    @property
    def kpts(self):
        return self._kpts

    @kpts.setter
    def kpts(self, kpts):
        if not hasattr(self, '_kpts') or kpts != self._kpts:
            self._kpts = kpts
            self.kpts_kc = monkhorst_pack(self.kpts)
            if hasattr(self, 'im_r'):
                del self.im_r  # we'll have to recalculate

    def calculate_energies_and_modes(self):
        if not self._already_read:
            if hasattr(self, 'im_r'):
                del self.im_r
            self.read()

        if not hasattr(self, 'im_r'):
            omega_kl, u_kl = self.vibrations.band_structure(
                self.kpts_kc, modes=True, verbose=self.verbose)

            self.im_r = self.vibrations.m_inv_x
            self.om_Q = omega_kl.ravel().real   # energies in eV
            self.modes_Qq = u_kl.reshape(len(self.om_Q),
                                         3 * len(self.atoms))
            self.modes_Qq /= self.im_r
            self.om_v = self.om_Q

            # pre-factors for one vibrational excitation
            with np.errstate(divide='ignore', invalid='ignore'):
                self.vib01_Q = np.where(
                    self.om_Q > 0, 1. / np.sqrt(2 * self.om_Q), 0)
            # -> sqrt(amu) * Angstrom
            self.vib01_Q *= np.sqrt(u.Ha * u._me / u._amu) * u.Bohr
