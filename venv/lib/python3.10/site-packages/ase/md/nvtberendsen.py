"""Berendsen NVT dynamics class."""

import warnings
from typing import Optional

import numpy as np

from ase import Atoms
from ase.md.md import MolecularDynamics


class NVTBerendsen(MolecularDynamics):
    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature: Optional[float] = None,
        taut: Optional[float] = None,
        fixcm: bool = True,
        *,
        temperature_K: Optional[float] = None,
        **kwargs,
    ):
        """Berendsen (constant N, V, T) molecular dynamics.

        Parameters
        ----------
        atoms: Atoms object
            The list of atoms.

        timestep: float
            The time step in ASE time units.

        temperature: float
            The desired temperature, in Kelvin.

        temperature_K: float
            Alias for *temperature*

        taut: float
            Time constant for Berendsen temperature coupling in ASE
            time units.

        fixcm: bool (optional)
            If True, the position and momentum of the center of mass is
            kept unperturbed.  Default: True.

        **kwargs : dict, optional
            Additional arguments passed to :class:~ase.md.md.MolecularDynamics
            base class.

        """
        if 'communicator' in kwargs:
            msg = (
                '`communicator` has been deprecated since ASE 3.25.0 '
                'and will be removed in ASE 3.26.0. Use `comm` instead.'
            )
            warnings.warn(msg, FutureWarning)
            kwargs['comm'] = kwargs.pop('communicator')

        MolecularDynamics.__init__(self, atoms, timestep, **kwargs)

        if taut is None:
            raise TypeError("Missing 'taut' argument.")
        self.taut = taut
        self.temperature = self._process_temperature(
            temperature, temperature_K, 'K'
        )

        self.fix_com = fixcm  # will the center of mass be held fixed?

    def set_taut(self, taut):
        self.taut = taut

    def get_taut(self):
        return self.taut

    def set_temperature(self, temperature=None, *, temperature_K=None):
        self.temperature = self._process_temperature(
            temperature, temperature_K, 'K'
        )

    def get_temperature(self):
        return self.temperature

    def set_timestep(self, timestep):
        self.dt = timestep

    def get_timestep(self):
        return self.dt

    def scale_velocities(self):
        """Do the NVT Berendsen velocity scaling"""
        tautscl = self.dt / self.taut
        old_temperature = self.atoms.get_temperature()

        scl_temperature = np.sqrt(
            1.0 + (self.temperature / old_temperature - 1.0) * tautscl
        )
        # Limit the velocity scaling to reasonable values
        if scl_temperature > 1.1:
            scl_temperature = 1.1
        if scl_temperature < 0.9:
            scl_temperature = 0.9

        p = self.atoms.get_momenta()
        p = scl_temperature * p
        self.atoms.set_momenta(p)
        return

    def step(self, forces=None):
        """Move one timestep forward using Berenden NVT molecular dynamics."""
        self.scale_velocities()

        # one step velocity verlet
        atoms = self.atoms

        if forces is None:
            forces = atoms.get_forces(md=True)

        p = self.atoms.get_momenta()
        p += 0.5 * self.dt * forces

        if self.fix_com:
            # calculate the center of mass
            # momentum and subtract it
            psum = p.sum(axis=0) / float(len(p))
            p = p - psum

        self.atoms.set_positions(
            self.atoms.get_positions()
            + self.dt * p / self.atoms.get_masses()[:, np.newaxis]
        )

        # We need to store the momenta on the atoms before calculating
        # the forces, as in a parallel Asap calculation atoms may
        # migrate during force calculations, and the momenta need to
        # migrate along with the atoms.  For the same reason, we
        # cannot use self.masses in the line above.

        self.atoms.set_momenta(p)
        forces = self.atoms.get_forces(md=True)
        atoms.set_momenta(self.atoms.get_momenta() + 0.5 * self.dt * forces)

        return forces
