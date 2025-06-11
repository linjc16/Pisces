# fmt: off

import numpy as np

import ase.units as u
from ase.calculators.excitation_list import polarizability
from ase.vibrations.raman import Raman, RamanPhonons
from ase.vibrations.resonant_raman import ResonantRaman


class Placzek(ResonantRaman):
    """Raman spectra within the Placzek approximation."""

    def __init__(self, *args, **kwargs):
        self._approx = 'PlaczekAlpha'
        ResonantRaman.__init__(self, *args, **kwargs)

    def set_approximation(self, value):
        raise ValueError('Approximation can not be set.')

    def _signed_disps(self, sign):
        for a, i in zip(self.myindices, self.myxyz):
            yield self._disp(a, i, sign)

    def _read_exobjs(self, sign):
        return [disp.read_exobj() for disp in self._signed_disps(sign)]

    def read_excitations(self):
        """Read excitations from files written"""
        self.ex0E_p = None  # mark as read
        self.exm_r = self._read_exobjs(sign=-1)
        self.exp_r = self._read_exobjs(sign=1)

    def electronic_me_Qcc(self, omega, gamma=0):
        self.calculate_energies_and_modes()

        V_rcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        pre = 1. / (2 * self.delta)
        pre *= u.Hartree * u.Bohr  # e^2Angstrom^2 / eV -> Angstrom^3

        om = omega
        if gamma:
            om += 1j * gamma

        for i, r in enumerate(self.myr):
            V_rcc[r] = pre * (
                polarizability(self.exp_r[i], om,
                               form=self.dipole_form, tensor=True) -
                polarizability(self.exm_r[i], om,
                               form=self.dipole_form, tensor=True))
        self.comm.sum(V_rcc)

        return self.map_to_modes(V_rcc)


class PlaczekStatic(Raman):
    def read_excitations(self):
        """Read excitations from files written"""
        self.al0_rr = None  # mark as read
        self.alm_rr = []
        self.alp_rr = []
        for a, i in zip(self.myindices, self.myxyz):
            for sign, al_rr in zip([-1, 1], [self.alm_rr, self.alp_rr]):
                disp = self._disp(a, i, sign)
                al_rr.append(disp.load_static_polarizability())

    def electronic_me_Qcc(self):
        self.calculate_energies_and_modes()

        V_rcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        pre = 1. / (2 * self.delta)
        pre *= u.Hartree * u.Bohr  # e^2Angstrom^2 / eV -> Angstrom^3

        for i, r in enumerate(self.myr):
            V_rcc[r] = pre * (self.alp_rr[i] - self.alm_rr[i])
        self.comm.sum(V_rcc)

        return self.map_to_modes(V_rcc)


class PlaczekStaticPhonons(RamanPhonons, PlaczekStatic):
    pass


class Profeta(ResonantRaman):
    """Profeta type approximations.

    Reference
    ---------
    Mickael Profeta and Francesco Mauri
    Phys. Rev. B 63 (2000) 245415
    """

    def __init__(self, *args, **kwargs):
        self.set_approximation(kwargs.pop('approximation', 'Profeta'))
        self.nonresonant = kwargs.pop('nonresonant', True)
        ResonantRaman.__init__(self, *args, **kwargs)

    def set_approximation(self, value):
        approx = value.lower()
        if approx in ['profeta', 'placzek', 'p-p']:
            self._approx = value
        else:
            raise ValueError('Please use "Profeta", "Placzek" or "P-P".')

    def electronic_me_profeta_rcc(self, omega, gamma=0.1,
                                  energy_derivative=False):
        """Raman spectra in Profeta and Mauri approximation

        Returns
        -------
        Electronic matrix element, unit Angstrom^2
         """
        self.calculate_energies_and_modes()

        V_rcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        pre = 1. / (2 * self.delta)
        pre *= u.Hartree * u.Bohr  # e^2Angstrom^2 / eV -> Angstrom^3

        def kappa_cc(me_pc, e_p, omega, gamma, form='v'):
            """Kappa tensor after Profeta and Mauri
            PRB 63 (2001) 245415"""
            k_cc = np.zeros((3, 3), dtype=complex)
            for p, me_c in enumerate(me_pc):
                me_cc = np.outer(me_c, me_c.conj())
                k_cc += me_cc / (e_p[p] - omega - 1j * gamma)
                if self.nonresonant:
                    k_cc += me_cc.conj() / (e_p[p] + omega + 1j * gamma)
            return k_cc

        mr = 0
        for a, i, r in zip(self.myindices, self.myxyz, self.myr):
            if energy_derivative >= 0:
                V_rcc[r] += pre * (
                    kappa_cc(self.expm_rpc[mr], self.ex0E_p,
                             omega, gamma, self.dipole_form) -
                    kappa_cc(self.exmm_rpc[mr], self.ex0E_p,
                             omega, gamma, self.dipole_form))
            if energy_derivative:
                V_rcc[r] += pre * (
                    kappa_cc(self.ex0m_pc, self.expE_rp[mr],
                             omega, gamma, self.dipole_form) -
                    kappa_cc(self.ex0m_pc, self.exmE_rp[mr],
                             omega, gamma, self.dipole_form))
            mr += 1
        self.comm.sum(V_rcc)

        return V_rcc

    def electronic_me_Qcc(self, omega, gamma):
        self.read()
        Vel_rcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        approximation = self.approximation.lower()
        if approximation == 'profeta':
            Vel_rcc += self.electronic_me_profeta_rcc(omega, gamma)
        elif approximation == 'placzek':
            Vel_rcc += self.electronic_me_profeta_rcc(omega, gamma, True)
        elif approximation == 'p-p':
            Vel_rcc += self.electronic_me_profeta_rcc(omega, gamma, -1)
        else:
            raise RuntimeError(
                'Bug: call with {} should not happen!'.format(
                    self.approximation))

        return self.map_to_modes(Vel_rcc)
