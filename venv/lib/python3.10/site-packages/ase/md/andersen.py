"""Andersen dynamics class."""

import warnings

from numpy import cos, log, ones, pi, random, repeat

from ase import Atoms, units
from ase.md.md import MolecularDynamics


class Andersen(MolecularDynamics):
    """Andersen (constant N, V, T) molecular dynamics."""

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        andersen_prob: float,
        fixcm: bool = True,
        rng=random,
        **kwargs,
    ):
        """
        Parameters
        ----------
        atoms: Atoms object
            The list of atoms.

        timestep: float
            The time step in ASE time units.

        temperature_K: float
            The desired temperature, in Kelvin.

        andersen_prob: float
            A random collision probability, typically 1e-4 to 1e-1.
            With this probability atoms get assigned random velocity components.

        fixcm: bool (optional)
            If True, the position and momentum of the center of mass is
            kept unperturbed.  Default: True.

        rng: RNG object, default: ``numpy.random``
            Random number generator. This must have the ``random`` method
            with the same signature as ``numpy.random.random``.

        **kwargs : dict, optional
            Additional arguments passed to :class:~ase.md.md.MolecularDynamics
            base class.

        The temperature is imposed by stochastic collisions with a heat bath
        that acts on velocity components of randomly chosen particles.
        The algorithm randomly decorrelates velocities, so dynamical properties
        like diffusion or viscosity cannot be properly measured.

        H. C. Andersen, J. Chem. Phys. 72 (4), 2384–2393 (1980)
        """
        if 'communicator' in kwargs:
            msg = (
                '`communicator` has been deprecated since ASE 3.25.0 '
                'and will be removed in ASE 3.26.0. Use `comm` instead.'
            )
            warnings.warn(msg, FutureWarning)
            kwargs['comm'] = kwargs.pop('communicator')
        self.temp = units.kB * temperature_K
        self.andersen_prob = andersen_prob
        self.fix_com = fixcm
        self.rng = rng
        MolecularDynamics.__init__(self, atoms, timestep, **kwargs)

    def set_temperature(self, temperature_K):
        self.temp = units.kB * temperature_K

    def set_andersen_prob(self, andersen_prob):
        self.andersen_prob = andersen_prob

    def set_timestep(self, timestep):
        self.dt = timestep

    def boltzmann_random(self, width, size):
        x = self.rng.random(size=size)
        y = self.rng.random(size=size)
        z = width * cos(2 * pi * x) * (-2 * log(1 - y)) ** 0.5
        return z

    def get_maxwell_boltzmann_velocities(self):
        natoms = len(self.atoms)
        masses = repeat(self.masses, 3).reshape(natoms, 3)
        width = (self.temp / masses) ** 0.5
        velos = self.boltzmann_random(width, size=(natoms, 3))
        return velos  # [[x, y, z],] components for each atom

    def step(self, forces=None):
        atoms = self.atoms

        if forces is None:
            forces = atoms.get_forces(md=True)

        self.v = atoms.get_velocities()

        # Random atom-wise variables are stored as attributes and broadcasted:
        #  - self.random_com_velocity  # added to all atoms if self.fix_com
        #  - self.random_velocity      # added to some atoms if the per-atom
        #  - self.andersen_chance      # andersen_chance <= andersen_prob
        # a dummy communicator will be used for serial runs

        if self.fix_com:
            # add random velocity to center of mass to prepare Andersen
            width = (self.temp / sum(self.masses)) ** 0.5
            self.random_com_velocity = ones(
                self.v.shape
            ) * self.boltzmann_random(width, (3))
            self.comm.broadcast(self.random_com_velocity, 0)
            self.v += self.random_com_velocity

        self.v += 0.5 * forces / self.masses * self.dt

        # apply Andersen thermostat
        self.random_velocity = self.get_maxwell_boltzmann_velocities()
        self.andersen_chance = self.rng.random(size=self.v.shape)
        self.comm.broadcast(self.random_velocity, 0)
        self.comm.broadcast(self.andersen_chance, 0)
        self.v[self.andersen_chance <= self.andersen_prob] = (
            self.random_velocity[self.andersen_chance <= self.andersen_prob]
        )

        x = atoms.get_positions()
        if self.fix_com:
            old_com = atoms.get_center_of_mass()
            self.v -= self._get_com_velocity(self.v)
        # Step: x^n -> x^(n+1) - this applies constraints if any
        atoms.set_positions(x + self.v * self.dt)
        if self.fix_com:
            atoms.set_center_of_mass(old_com)

        # recalc velocities after RATTLE constraints are applied
        self.v = (atoms.get_positions() - x) / self.dt
        forces = atoms.get_forces(md=True)

        # Update the velocities
        self.v += 0.5 * forces / self.masses * self.dt

        if self.fix_com:
            self.v -= self._get_com_velocity(self.v)

        # Second part of RATTLE taken care of here
        atoms.set_momenta(self.v * self.masses)

        return forces
