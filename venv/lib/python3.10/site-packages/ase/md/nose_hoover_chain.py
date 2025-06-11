# fmt: off

from __future__ import annotations

import numpy as np
from scipy.special import exprel

import ase.units
from ase import Atoms
from ase.md.md import MolecularDynamics

# Coefficients for the fourth-order Suzuki-Yoshida integration scheme
# Ref: H. Yoshida, Phys. Lett. A 150, 5-7, 262-268 (1990).
#      https://doi.org/10.1016/0375-9601(90)90092-3
FOURTH_ORDER_COEFFS = [
    1 / (2 - 2 ** (1 / 3)),
    -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)),
    1 / (2 - 2 ** (1 / 3)),
]


class NoseHooverChainNVT(MolecularDynamics):
    """Isothermal molecular dynamics with Nose-Hoover chain.

    This implementation is based on the Nose-Hoover chain equations and
    the Liouville-operator derived integrator for non-Hamiltonian systems [1-3].

    - [1] G. J. Martyna, M. L. Klein, and M. E. Tuckerman, J. Chem. Phys. 97,
          2635 (1992). https://doi.org/10.1063/1.463940
    - [2] M. E. Tuckerman, J. Alejandre, R. López-Rendón, A. L. Jochim,
          and G. J. Martyna, J. Phys. A: Math. Gen. 39, 5629 (2006).
          https://doi.org/10.1088/0305-4470/39/19/S18
    - [3] M. E. Tuckerman, Statistical Mechanics: Theory and Molecular
          Simulation, Oxford University Press (2010).

    While the algorithm and notation for the thermostat are largely adapted
    from Ref. [4], the core equations are detailed in the implementation
    note available at
    https://github.com/lan496/lan496.github.io/blob/main/notes/nose_hoover_chain/main.pdf.

    - [4] M. E. Tuckerman, Statistical Mechanics: Theory and Molecular
          Simulation, 2nd ed. (Oxford University Press, 2009).
    """

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        tdamp: float,
        tchain: int = 3,
        tloop: int = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        atoms: ase.Atoms
            The atoms object.
        timestep: float
            The time step in ASE time units.
        temperature_K: float
            The target temperature in K.
        tdamp: float
            The characteristic time scale for the thermostat in ASE time units.
            Typically, it is set to 100 times of `timestep`.
        tchain: int
            The number of thermostat variables in the Nose-Hoover chain.
        tloop: int
            The number of sub-steps in thermostat integration.
        **kwargs : dict, optional
            Additional arguments passed to :class:~ase.md.md.MolecularDynamics
            base class.
        """
        super().__init__(
            atoms=atoms,
            timestep=timestep,
            **kwargs,
        )
        assert self.masses.shape == (len(self.atoms), 1)

        num_atoms = self.atoms.get_global_number_of_atoms()
        self._thermostat = NoseHooverChainThermostat(
            num_atoms_global=num_atoms,
            masses=self.masses,
            temperature_K=temperature_K,
            tdamp=tdamp,
            tchain=tchain,
            tloop=tloop,
        )

        # The following variables are updated during self.step()
        self._q = self.atoms.get_positions()
        self._p = self.atoms.get_momenta()

    def step(self) -> None:
        dt2 = self.dt / 2
        self._p = self._thermostat.integrate_nhc(self._p, dt2)
        self._integrate_p(dt2)
        self._integrate_q(self.dt)
        self._integrate_p(dt2)
        self._p = self._thermostat.integrate_nhc(self._p, dt2)

        self._update_atoms()

    def get_conserved_energy(self) -> float:
        """Return the conserved energy-like quantity.

        This method is mainly used for testing.
        """
        conserved_energy = (
            self.atoms.get_potential_energy(force_consistent=True)
            + self.atoms.get_kinetic_energy()
            + self._thermostat.get_thermostat_energy()
        )
        return float(conserved_energy)

    def _update_atoms(self) -> None:
        self.atoms.set_positions(self._q)
        self.atoms.set_momenta(self._p)

    def _get_forces(self) -> np.ndarray:
        self._update_atoms()
        return self.atoms.get_forces(md=True)

    def _integrate_q(self, delta: float) -> None:
        """Integrate exp(i * L_1 * delta)"""
        self._q += delta * self._p / self.masses

    def _integrate_p(self, delta: float) -> None:
        """Integrate exp(i * L_2 * delta)"""
        forces = self._get_forces()
        self._p += delta * forces


class NoseHooverChainThermostat:
    """Nose-Hoover chain style thermostats.

    See `NoseHooverChainNVT` for the references.
    """
    def __init__(
        self,
        num_atoms_global: int,
        masses: np.ndarray,
        temperature_K: float,
        tdamp: float,
        tchain: int = 3,
        tloop: int = 1,
    ):
        """See `NoseHooverChainNVT` for the parameters."""
        self._num_atoms_global = num_atoms_global
        self._masses = masses  # (len(atoms), 1)
        self._tdamp = tdamp
        self._tchain = tchain
        self._tloop = tloop

        self._kT = ase.units.kB * temperature_K

        assert tchain >= 1
        self._Q = np.zeros(tchain)
        self._Q[0] = 3 * self._num_atoms_global * self._kT * tdamp**2
        self._Q[1:] = self._kT * tdamp**2

        # The following variables are updated during self.step()
        self._eta = np.zeros(self._tchain)
        self._p_eta = np.zeros(self._tchain)

    def get_thermostat_energy(self) -> float:
        """Return energy-like contribution from the thermostat variables."""
        energy = (
            3 * self._num_atoms_global * self._kT * self._eta[0]
            + self._kT * np.sum(self._eta[1:])
            + np.sum(0.5 * self._p_eta**2 / self._Q)
        )
        return float(energy)

    def integrate_nhc(self, p: np.ndarray, delta: float) -> np.ndarray:
        """Integrate exp(i * L_NHC * delta) and update momenta `p`."""
        for _ in range(self._tloop):
            for coeff in FOURTH_ORDER_COEFFS:
                p = self._integrate_nhc_loop(
                    p, coeff * delta / len(FOURTH_ORDER_COEFFS)
                )

        return p

    def _integrate_p_eta_j(self, p: np.ndarray, j: int,
                           delta2: float, delta4: float) -> None:
        if j < self._tchain - 1:
            self._p_eta[j] *= np.exp(
                -delta4 * self._p_eta[j + 1] / self._Q[j + 1]
            )

        if j == 0:
            g_j = np.sum(p**2 / self._masses) \
                - 3 * self._num_atoms_global * self._kT
        else:
            g_j = self._p_eta[j - 1] ** 2 / self._Q[j - 1] - self._kT
        self._p_eta[j] += delta2 * g_j

        if j < self._tchain - 1:
            self._p_eta[j] *= np.exp(
                -delta4 * self._p_eta[j + 1] / self._Q[j + 1]
            )

    def _integrate_eta(self, delta: float) -> None:
        self._eta += delta * self._p_eta / self._Q

    def _integrate_nhc_p(self, p: np.ndarray, delta: float) -> None:
        p *= np.exp(-delta * self._p_eta[0] / self._Q[0])

    def _integrate_nhc_loop(self, p: np.ndarray, delta: float) -> np.ndarray:
        delta2 = delta / 2
        delta4 = delta / 4

        for j in range(self._tchain):
            self._integrate_p_eta_j(p, self._tchain - j - 1, delta2, delta4)
        self._integrate_eta(delta)
        self._integrate_nhc_p(p, delta)
        for j in range(self._tchain):
            self._integrate_p_eta_j(p, j, delta2, delta4)

        return p


class IsotropicMTKNPT(MolecularDynamics):
    """Isothermal-isobaric molecular dynamics with isotropic volume fluctuations
    by Martyna-Tobias-Klein (MTK) method [1].

    See also `NoseHooverChainNVT` for the references.

    - [1] G. J. Martyna, D. J. Tobias, and M. L. Klein, J. Chem. Phys. 101,
          4177-4189 (1994). https://doi.org/10.1063/1.467468
    """
    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        pressure_au: float,
        tdamp: float,
        pdamp: float,
        tchain: int = 3,
        pchain: int = 3,
        tloop: int = 1,
        ploop: int = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        atoms: ase.Atoms
            The atoms object.
        timestep: float
            The time step in ASE time units.
        temperature_K: float
            The target temperature in K.
        pressure_au: float
            The external pressure in eV/Ang^3.
        tdamp: float
            The characteristic time scale for the thermostat in ASE time units.
            Typically, it is set to 100 times of `timestep`.
        pdamp: float
            The characteristic time scale for the barostat in ASE time units.
            Typically, it is set to 1000 times of `timestep`.
        tchain: int
            The number of thermostat variables in the Nose-Hoover thermostat.
        pchain: int
            The number of barostat variables in the MTK barostat.
        tloop: int
            The number of sub-steps in thermostat integration.
        ploop: int
            The number of sub-steps in barostat integration.
        **kwargs : dict, optional
            Additional arguments passed to :class:~ase.md.md.MolecularDynamics
            base class.
        """
        super().__init__(
            atoms=atoms,
            timestep=timestep,
            **kwargs,
        )
        assert self.masses.shape == (len(self.atoms), 1)

        if len(atoms.constraints) > 0:
            raise NotImplementedError(
                "Current implementation does not support constraints"
            )

        self._num_atoms_global = self.atoms.get_global_number_of_atoms()
        self._thermostat = NoseHooverChainThermostat(
            num_atoms_global=self._num_atoms_global,
            masses=self.masses,
            temperature_K=temperature_K,
            tdamp=tdamp,
            tchain=tchain,
            tloop=tloop,
        )
        self._barostat = IsotropicMTKBarostat(
            num_atoms_global=self._num_atoms_global,
            masses=self.masses,
            temperature_K=temperature_K,
            pdamp=pdamp,
            pchain=pchain,
            ploop=ploop,
        )

        self._temperature_K = temperature_K
        self._pressure_au = pressure_au

        self._kT = ase.units.kB * self._temperature_K
        self._volume0 = self.atoms.get_volume()
        self._cell0 = np.array(self.atoms.get_cell())

        # The following variables are updated during self.step()
        self._q = self.atoms.get_positions()  # positions
        self._p = self.atoms.get_momenta()  # momenta
        self._eps = 0.0  # volume
        self._p_eps = 0.0  # volume momenta

    def step(self) -> None:
        dt2 = self.dt / 2

        self._p_eps = self._barostat.integrate_nhc_baro(self._p_eps, dt2)
        self._p = self._thermostat.integrate_nhc(self._p, dt2)
        self._integrate_p_cell(dt2)
        self._integrate_p(dt2)
        self._integrate_q(self.dt)
        self._integrate_q_cell(self.dt)
        self._integrate_p(dt2)
        self._integrate_p_cell(dt2)
        self._p = self._thermostat.integrate_nhc(self._p, dt2)
        self._p_eps = self._barostat.integrate_nhc_baro(self._p_eps, dt2)

        self._update_atoms()

    def get_conserved_energy(self) -> float:
        """Return the conserved energy-like quantity.

        This method is mainly used for testing.
        """
        conserved_energy = (
            self.atoms.get_potential_energy(force_consistent=True)
            + self.atoms.get_kinetic_energy()
            + self._thermostat.get_thermostat_energy()
            + self._barostat.get_barostat_energy()
            + self._p_eps * self._p_eps / (2 * self._barostat.W)
            + self._pressure_au * self._get_volume()
        )
        return float(conserved_energy)

    def _update_atoms(self) -> None:
        self.atoms.set_positions(self._q)
        self.atoms.set_momenta(self._p)
        cell = self._cell0 * np.exp(self._eps)
        # Never set scale_atoms=True
        self.atoms.set_cell(cell, scale_atoms=False)

    def _get_volume(self) -> float:
        return self._volume0 * np.exp(3 * self._eps)

    def _get_forces(self) -> np.ndarray:
        self._update_atoms()
        return self.atoms.get_forces(md=True)

    def _get_pressure(self) -> np.ndarray:
        self._update_atoms()
        stress = self.atoms.get_stress(voigt=False, include_ideal_gas=True)
        pressure = -np.trace(stress) / 3
        return pressure

    def _integrate_q(self, delta: float) -> None:
        """Integrate exp(i * L_1 * delta)"""
        x = delta * self._p_eps / self._barostat.W
        self._q = (
            self._q * np.exp(x)
            + self._p * delta / self.masses * exprel(x)
        )

    def _integrate_p(self, delta: float) -> None:
        """Integrate exp(i * L_2 * delta)"""
        x = (1 + 1 / self._num_atoms_global) * self._p_eps * delta \
                / self._barostat.W
        forces = self._get_forces()
        self._p = self._p * np.exp(-x) + delta * forces * exprel(-x)

    def _integrate_q_cell(self, delta: float) -> None:
        """Integrate exp(i * L_(epsilon, 1) * delta)"""
        self._eps += delta * self._p_eps / self._barostat.W

    def _integrate_p_cell(self, delta: float) -> None:
        """Integrate exp(i * L_(epsilon, 2) * delta)"""
        pressure = self._get_pressure()
        volume = self._get_volume()
        G = (
            3 * volume * (pressure - self._pressure_au)
            + np.sum(self._p**2 / self.masses) / self._num_atoms_global
        )
        self._p_eps += delta * G


class IsotropicMTKBarostat:
    """MTK barostat for isotropic volume fluctuations.

    See `IsotropicMTKNPT` for the references.
    """
    def __init__(
        self,
        num_atoms_global: int,
        masses: np.ndarray,
        temperature_K: float,
        pdamp: float,
        pchain: int = 3,
        ploop: int = 1,
    ):
        self._num_atoms_global = num_atoms_global
        self._masses = masses  # (num_atoms, 1)
        self._pdamp = pdamp
        self._pchain = pchain
        self._ploop = ploop

        self._kT = ase.units.kB * temperature_K

        self._W = (3 * self._num_atoms_global + 3) * self._kT * self._pdamp**2

        assert pchain >= 1
        self._R = np.zeros(self._pchain)
        self._R[0] = 9 * self._kT * self._pdamp**2
        self._R[1:] = self._kT * self._pdamp**2

        self._xi = np.zeros(self._pchain)  # barostat coordinates
        self._p_xi = np.zeros(self._pchain)

    @property
    def W(self) -> float:
        """Virtual mass for barostat momenta `p_xi`."""
        return self._W

    def get_barostat_energy(self) -> float:
        """Return energy-like contribution from the barostat variables."""
        energy = (
            + np.sum(0.5 * self._p_xi**2 / self._R)
            + self._kT * np.sum(self._xi)
        )
        return float(energy)

    def integrate_nhc_baro(self, p_eps: float, delta: float) -> float:
        """Integrate exp(i * L_NHC-baro * delta)"""
        for _ in range(self._ploop):
            for coeff in FOURTH_ORDER_COEFFS:
                p_eps = self._integrate_nhc_baro_loop(
                    p_eps, coeff * delta / len(FOURTH_ORDER_COEFFS)
                )
        return p_eps

    def _integrate_nhc_baro_loop(self, p_eps: float, delta: float) -> float:
        delta2 = delta / 2
        delta4 = delta / 4

        for j in range(self._pchain):
            self._integrate_p_xi_j(p_eps, self._pchain - j - 1, delta2, delta4)
        self._integrate_xi(delta)
        p_eps = self._integrate_nhc_p_eps(p_eps, delta)
        for j in range(self._pchain):
            self._integrate_p_xi_j(p_eps, j, delta2, delta4)

        return p_eps

    def _integrate_p_xi_j(self, p_eps: float, j: int,
                          delta2: float, delta4: float) -> None:
        if j < self._pchain - 1:
            self._p_xi[j] *= np.exp(
                -delta4 * self._p_xi[j + 1] / self._R[j + 1]
            )

        if j == 0:
            g_j = p_eps ** 2 / self._W - self._kT
        else:
            g_j = self._p_xi[j - 1] ** 2 / self._R[j - 1] - self._kT
        self._p_xi[j] += delta2 * g_j

        if j < self._pchain - 1:
            self._p_xi[j] *= np.exp(
                -delta4 * self._p_xi[j + 1] / self._R[j + 1]
            )

    def _integrate_xi(self, delta) -> None:
        self._xi += delta * self._p_xi / self._R

    def _integrate_nhc_p_eps(self, p_eps: float, delta: float) -> float:
        p_eps_new = p_eps * float(
            np.exp(-delta * self._p_xi[0] / self._R[0])
        )
        return p_eps_new
