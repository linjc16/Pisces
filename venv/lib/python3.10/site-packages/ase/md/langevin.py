"""Langevin dynamics class."""

import warnings
from typing import Optional

import numpy as np

from ase import Atoms, units
from ase.md.md import MolecularDynamics


class Langevin(MolecularDynamics):
    """Langevin (constant N, V, T) molecular dynamics."""

    # Helps Asap doing the right thing.  Increment when changing stuff:
    _lgv_version = 5

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature: Optional[float] = None,
        friction: Optional[float] = None,
        fixcm: bool = True,
        *,
        temperature_K: Optional[float] = None,
        rng=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        atoms: Atoms object
            The list of atoms.

        timestep: float
            The time step in ASE time units.

        temperature: float (deprecated)
            The desired temperature, in electron volt.

        temperature_K: float
            The desired temperature, in Kelvin.

        friction: float
            A friction coefficient in inverse ASE time units.
            For example, set ``0.01 / ase.units.fs`` to provide
            0.01 fs\\ :sup:`−1` (10 ps\\ :sup:`−1`).

        fixcm: bool (optional)
            If True, the position and momentum of the center of mass is
            kept unperturbed.  Default: True.

        rng: RNG object (optional)
            Random number generator, by default numpy.random.  Must have a
            standard_normal method matching the signature of
            numpy.random.standard_normal.

        **kwargs : dict, optional
            Additional arguments passed to :class:~ase.md.md.MolecularDynamics
            base class.

        The temperature and friction are normally scalars, but in principle one
        quantity per atom could be specified by giving an array.

        RATTLE constraints can be used with these propagators, see:
        E. V.-Eijnden, and G. Ciccotti, Chem. Phys. Lett. 429, 310 (2006)

        The propagator is Equation 23 (Eq. 39 if RATTLE constraints are used)
        of the above reference.  That reference also contains another
        propagator in Eq. 21/34; but that propagator is not quasi-symplectic
        and gives a systematic offset in the temperature at large time steps.
        """
        if 'communicator' in kwargs:
            msg = (
                '`communicator` has been deprecated since ASE 3.25.0 '
                'and will be removed in ASE 3.26.0. Use `comm` instead.'
            )
            warnings.warn(msg, FutureWarning)
            kwargs['comm'] = kwargs.pop('communicator')

        if friction is None:
            raise TypeError("Missing 'friction' argument.")
        self.fr = friction
        self.temp = units.kB * self._process_temperature(
            temperature, temperature_K, 'eV'
        )
        self.fix_com = fixcm

        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng
        MolecularDynamics.__init__(self, atoms, timestep, **kwargs)
        self.updatevars()

    def todict(self):
        d = MolecularDynamics.todict(self)
        d.update(
            {
                'temperature_K': self.temp / units.kB,
                'friction': self.fr,
                'fixcm': self.fix_com,
            }
        )
        return d

    def set_temperature(self, temperature=None, temperature_K=None):
        self.temp = units.kB * self._process_temperature(
            temperature, temperature_K, 'eV'
        )
        self.updatevars()

    def set_friction(self, friction):
        self.fr = friction
        self.updatevars()

    def set_timestep(self, timestep):
        self.dt = timestep
        self.updatevars()

    def updatevars(self):
        dt = self.dt
        T = self.temp
        fr = self.fr
        masses = self.masses
        sigma = np.sqrt(2 * T * fr / masses)

        self.c1 = dt / 2.0 - dt * dt * fr / 8.0
        self.c2 = dt * fr / 2 - dt * dt * fr * fr / 8.0
        self.c3 = np.sqrt(dt) * sigma / 2.0 - dt**1.5 * fr * sigma / 8.0
        self.c5 = dt**1.5 * sigma / (2 * np.sqrt(3))
        self.c4 = fr / 2.0 * self.c5

    def step(self, forces=None):
        atoms = self.atoms
        natoms = len(atoms)

        if forces is None:
            forces = atoms.get_forces(md=True)

        # This velocity as well as rnd_pos, rnd_mom and a few other
        # variables are stored as attributes, so Asap can do its magic
        # when atoms migrate between processors.
        self.v = atoms.get_velocities()

        xi = self.rng.standard_normal(size=(natoms, 3))
        eta = self.rng.standard_normal(size=(natoms, 3))

        # When holonomic constraints for rigid linear triatomic molecules are
        # present, ask the constraints to redistribute xi and eta within each
        # triple defined in the constraints. This is needed to achieve the
        # correct target temperature.
        for constraint in self.atoms.constraints:
            if hasattr(constraint, 'redistribute_forces_md'):
                constraint.redistribute_forces_md(atoms, xi, rand=True)
                constraint.redistribute_forces_md(atoms, eta, rand=True)

        self.comm.broadcast(xi, 0)
        self.comm.broadcast(eta, 0)

        # To keep the center of mass stationary, we have to calculate
        # the random perturbations to the positions and the momenta,
        # and make sure that they sum to zero.  This perturbs the
        # temperature slightly, and we have to correct.
        self.rnd_pos = self.c5 * eta
        self.rnd_vel = self.c3 * xi - self.c4 * eta
        if self.fix_com:
            factor = np.sqrt(natoms / (natoms - 1.0))
            self.rnd_pos -= self.rnd_pos.sum(axis=0) / natoms
            self.rnd_vel -= (self.rnd_vel * self.masses).sum(axis=0) / (
                self.masses * natoms
            )
            self.rnd_pos *= factor
            self.rnd_vel *= factor

        # First halfstep in the velocity.
        self.v += (
            self.c1 * forces / self.masses - self.c2 * self.v + self.rnd_vel
        )

        # Full step in positions
        x = atoms.get_positions()

        # Step: x^n -> x^(n+1) - this applies constraints if any.
        atoms.set_positions(x + self.dt * self.v + self.rnd_pos)

        # recalc velocities after RATTLE constraints are applied
        self.v = (self.atoms.get_positions() - x - self.rnd_pos) / self.dt
        forces = atoms.get_forces(md=True)

        # Update the velocities
        self.v += (
            self.c1 * forces / self.masses - self.c2 * self.v + self.rnd_vel
        )

        # Second part of RATTLE taken care of here
        atoms.set_momenta(self.v * self.masses)

        return forces
