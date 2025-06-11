# fmt: off

# VelocityDistributions.py -- set up a velocity distribution

"""Module for setting up velocity distributions such as Maxwell–Boltzmann.

Currently, only a few functions are defined, such as
MaxwellBoltzmannDistribution, which sets the momenta of a list of
atoms according to a Maxwell-Boltzmann distribution at a given
temperature.
"""
from typing import Literal, Optional

import numpy as np

from ase import Atoms, units
from ase.md.md import process_temperature
from ase.parallel import world

# define a ``zero'' temperature to avoid divisions by zero
eps_temp = 1e-12


class UnitError(Exception):
    """Exception raised when wrong units are specified"""


def force_temperature(atoms: Atoms,
                      temperature: float,
                      unit: Literal["K", "eV"] = "K"):
    """
    Force the temperature of the atomic system to a precise value.

    This function adjusts the momenta of the atoms to achieve the
    exact specified temperature, overriding the Maxwell-Boltzmann
    distribution. The temperature can be specified in either Kelvin
    (K) or electron volts (eV).

    Parameters
    ----------
    atoms
        The atomic system represented as an ASE Atoms object.
    temperature
        The exact temperature to force for the atomic system.
    unit
        The unit of the specified temperature.
        Can be either 'K' (Kelvin) or 'eV' (electron volts). Default is 'K'.
    """

    if unit == "K":
        target_temp = temperature * units.kB
    elif unit == "eV":
        target_temp = temperature
    else:
        raise UnitError(f"'{unit}' is not supported, use 'K' or 'eV'.")

    if temperature > eps_temp:
        ndof = atoms.get_number_of_degrees_of_freedom()
        current_temp = 2 * atoms.get_kinetic_energy() / ndof
        scale = target_temp / current_temp
    else:
        scale = 0.0

    atoms.set_momenta(atoms.get_momenta() * np.sqrt(scale))


def _maxwellboltzmanndistribution(masses, temp, communicator=None, rng=None):
    """Return a Maxwell-Boltzmann distribution with a given temperature.

    Paremeters:

    masses: float
        The atomic masses.

    temp: float
        The temperature in electron volt.

    communicator: MPI communicator (optional)
        Communicator used to distribute an identical distribution to
        all tasks.  Set to 'serial' to disable communication (setting to None
        gives the default).  Default: ase.parallel.world

    rng: numpy RNG (optional)
        The random number generator.  Default: np.random

    Returns:

    A numpy array with Maxwell-Boltzmann distributed momenta.
    """
    if rng is None:
        rng = np.random
    if communicator is None:
        communicator = world
    xi = rng.standard_normal((len(masses), 3))
    if communicator != 'serial':
        communicator.broadcast(xi, 0)
    momenta = xi * np.sqrt(masses * temp)[:, np.newaxis]
    return momenta


def MaxwellBoltzmannDistribution(
    atoms: Atoms,
    temp: Optional[float] = None,
    *,
    temperature_K: Optional[float] = None,
    communicator=None,
    force_temp: bool = False,
    rng=None,
):
    """Set the atomic momenta to a Maxwell-Boltzmann distribution.

    Parameters:

    atoms: Atoms object
        The atoms.  Their momenta will be modified.

    temp: float (deprecated)
        The temperature in eV.  Deprecated, use ``temperature_K`` instead.

    temperature_K: float
        The temperature in Kelvin.

    communicator: MPI communicator (optional)
        Communicator used to distribute an identical distribution to
        all tasks.  Set to 'serial' to disable communication.  Leave as None to
        get the default: ase.parallel.world

    force_temp: bool (optional, default: False)
        If True, the random momenta are rescaled so the kinetic energy is
        exactly 3/2 N k T.  This is a slight deviation from the correct
        Maxwell-Boltzmann distribution.

    rng: Numpy RNG (optional)
        Random number generator.  Default: numpy.random
        If you would like to always get the identical distribution, you can
        supply a random seed like ``rng=numpy.random.RandomState(seed)``, where
        seed is an integer.
    """
    temp = units.kB * process_temperature(temp, temperature_K, 'eV')
    masses = atoms.get_masses()
    momenta = _maxwellboltzmanndistribution(masses, temp, communicator, rng)
    atoms.set_momenta(momenta)
    if force_temp:
        force_temperature(atoms, temperature=temp, unit="eV")


def Stationary(atoms: Atoms, preserve_temperature: bool = True):
    "Sets the center-of-mass momentum to zero."

    # Save initial temperature
    temp0 = atoms.get_temperature()

    p = atoms.get_momenta()
    p0 = np.sum(p, 0)
    # We should add a constant velocity, not momentum, to the atoms
    m = atoms.get_masses()
    mtot = np.sum(m)
    v0 = p0 / mtot
    p -= v0 * m[:, np.newaxis]
    atoms.set_momenta(p)

    if preserve_temperature:
        force_temperature(atoms, temp0)


def ZeroRotation(atoms: Atoms, preserve_temperature: float = True):
    "Sets the total angular momentum to zero by counteracting rigid rotations."

    # Save initial temperature
    temp0 = atoms.get_temperature()

    # Find the principal moments of inertia and principal axes basis vectors
    Ip, basis = atoms.get_moments_of_inertia(vectors=True)
    # Calculate the total angular momentum and transform to principal basis
    Lp = np.dot(basis, atoms.get_angular_momentum())
    # Calculate the rotation velocity vector in the principal basis, avoiding
    # zero division, and transform it back to the cartesian coordinate system
    omega = np.dot(np.linalg.inv(basis), np.select([Ip > 0], [Lp / Ip]))
    # We subtract a rigid rotation corresponding to this rotation vector
    com = atoms.get_center_of_mass()
    positions = atoms.get_positions()
    positions -= com  # translate center of mass to origin
    velocities = atoms.get_velocities()
    atoms.set_velocities(velocities - np.cross(omega, positions))

    if preserve_temperature:
        force_temperature(atoms, temp0)


def n_BE(temp, omega):
    """Bose-Einstein distribution function.

    Args:
        temp: temperature converted to eV (*units.kB)
        omega: sequence of frequencies converted to eV

    Returns:
        Value of Bose-Einstein distribution function for each energy

    """

    omega = np.asarray(omega)

    # 0K limit
    if temp < eps_temp:
        n = np.zeros_like(omega)
    else:
        n = 1 / (np.exp(omega / (temp)) - 1)
    return n


def phonon_harmonics(
    force_constants,
    masses,
    temp=None,
    *,
    temperature_K=None,
    rng=np.random.rand,
    quantum=False,
    plus_minus=False,
    return_eigensolution=False,
    failfast=True,
):
    r"""Return displacements and velocities that produce a given temperature.

    Parameters:

    force_constants: array of size 3N x 3N
        force constants (Hessian) of the system in eV/Å²

    masses: array of length N
        masses of the structure in amu

    temp: float (deprecated)
        Temperature converted to eV (T * units.kB).  Deprecated, use
        ``temperature_K``.

    temperature_K: float
        Temperature in Kelvin.

    rng: function
        Random number generator function, e.g., np.random.rand

    quantum: bool
        True for Bose-Einstein distribution, False for Maxwell-Boltzmann
        (classical limit)

    plus_minus: bool
        Displace atoms with +/- the amplitude accoding to PRB 94, 075125

    return_eigensolution: bool
        return eigenvalues and eigenvectors of the dynamical matrix

    failfast: bool
        True for sanity checking the phonon spectrum for negative
        frequencies at Gamma

    Returns:

    Displacements, velocities generated from the eigenmodes,
    (optional: eigenvalues, eigenvectors of dynamical matrix)

    Purpose:

    Excite phonon modes to specified temperature.

    This excites all phonon modes randomly so that each contributes,
    on average, equally to the given temperature.  Both potential
    energy and kinetic energy will be consistent with the phononic
    vibrations characteristic of the specified temperature.

    In other words the system will be equilibrated for an MD run at
    that temperature.

    force_constants should be the matrix as force constants, e.g.,
    as computed by the ase.phonons module.

    Let X_ai be the phonon modes indexed by atom and mode, w_i the
    phonon frequencies, and let 0 < Q_i <= 1 and 0 <= R_i < 1 be
    uniformly random numbers.  Then

    .. code-block:: none


                    1/2
       _     / k T \     ---  1  _             1/2
       R  += | --- |      >  --- X   (-2 ln Q )    cos (2 pi R )
        a    \  m  /     ---  w   ai         i                i
                 a        i    i


                    1/2
       _     / k T \     --- _            1/2
       v   = | --- |      >  X  (-2 ln Q )    sin (2 pi R )
        a    \  m  /     ---  ai        i                i
                 a        i

    Reference: [West, Estreicher; PRL 96, 22 (2006)]
    """

    # Handle the temperature units
    temp = units.kB * process_temperature(temp, temperature_K, 'eV')

    # Build dynamical matrix
    rminv = (masses ** -0.5).repeat(3)
    dynamical_matrix = force_constants * rminv[:, None] * rminv[None, :]

    # Solve eigenvalue problem to compute phonon spectrum and eigenvectors
    w2_s, X_is = np.linalg.eigh(dynamical_matrix)

    # Check for soft modes
    if failfast:
        zeros = w2_s[:3]
        worst_zero = np.abs(zeros).max()
        if worst_zero > 1e-3:
            msg = "Translational deviate from 0 significantly: "
            raise ValueError(msg + f"{w2_s[:3]}")

        w2min = w2_s[3:].min()
        if w2min < 0:
            msg = "Dynamical matrix has negative eigenvalues such as "
            raise ValueError(msg + f"{w2min}")

    # First three modes are translational so ignore:
    nw = len(w2_s) - 3
    n_atoms = len(masses)
    w_s = np.sqrt(w2_s[3:])
    X_acs = X_is[:, 3:].reshape(n_atoms, 3, nw)

    # Assign the amplitudes according to Bose-Einstein distribution
    # or high temperature (== classical) limit
    if quantum:
        hbar = units._hbar * units.J * units.s
        A_s = np.sqrt(hbar * (2 * n_BE(temp, hbar * w_s) + 1) / (2 * w_s))
    else:
        A_s = np.sqrt(temp) / w_s

    if plus_minus:
        # create samples by multiplying the amplitude with +/-
        # according to Eq. 5 in PRB 94, 075125

        spread = (-1) ** np.arange(nw)

        # gauge eigenvectors: largest value always positive
        for ii in range(X_acs.shape[-1]):
            vec = X_acs[:, :, ii]
            max_arg = np.argmax(abs(vec))
            X_acs[:, :, ii] *= np.sign(vec.flat[max_arg])

        # Create velocities und displacements from the amplitudes and
        # eigenvectors
        A_s *= spread
        phi_s = 2.0 * np.pi * rng(nw)

        # Assign velocities, sqrt(2) compensates for missing sin(phi) in
        # amplitude for displacement
        v_ac = (w_s * A_s * np.sqrt(2) * np.cos(phi_s) * X_acs).sum(axis=2)
        v_ac /= np.sqrt(masses)[:, None]

        # Assign displacements
        d_ac = (A_s * X_acs).sum(axis=2)
        d_ac /= np.sqrt(masses)[:, None]

    else:
        # compute the gaussian distribution for the amplitudes
        # We need 0 < P <= 1.0 and not 0 0 <= P < 1.0 for the logarithm
        # to avoid (highly improbable) NaN.

        # Box Muller [en.wikipedia.org/wiki/Box–Muller_transform]:
        spread = np.sqrt(-2.0 * np.log(1.0 - rng(nw)))

        # assign amplitudes and phases
        A_s *= spread
        phi_s = 2.0 * np.pi * rng(nw)

        # Assign velocities and displacements
        v_ac = (w_s * A_s * np.cos(phi_s) * X_acs).sum(axis=2)
        v_ac /= np.sqrt(masses)[:, None]

        d_ac = (A_s * np.sin(phi_s) * X_acs).sum(axis=2)
        d_ac /= np.sqrt(masses)[:, None]

    if return_eigensolution:
        return d_ac, v_ac, w2_s, X_is
    # else
    return d_ac, v_ac


def PhononHarmonics(
    atoms,
    force_constants,
    temp=None,
    *,
    temperature_K=None,
    rng=np.random,
    quantum=False,
    plus_minus=False,
    return_eigensolution=False,
    failfast=True,
):
    r"""Excite phonon modes to specified temperature.

    This will displace atomic positions and set the velocities so as
    to produce a random, phononically correct state with the requested
    temperature.

    Parameters:

    atoms: ase.atoms.Atoms() object
        Positions and momenta of this object are perturbed.

    force_constants: ndarray of size 3N x 3N
        Force constants for the the structure represented by atoms in eV/Å²

    temp: float (deprecated).
        Temperature in eV.  Deprecated, use ``temperature_K`` instead.

    temperature_K: float
        Temperature in Kelvin.

    rng: Random number generator
        RandomState or other random number generator, e.g., np.random.rand

    quantum: bool
        True for Bose-Einstein distribution, False for Maxwell-Boltzmann
        (classical limit)

    failfast: bool
        True for sanity checking the phonon spectrum for negative frequencies
        at Gamma.

    """

    # Receive displacements and velocities from phonon_harmonics()
    d_ac, v_ac = phonon_harmonics(
        force_constants=force_constants,
        masses=atoms.get_masses(),
        temp=temp,
        temperature_K=temperature_K,
        rng=rng.rand,
        plus_minus=plus_minus,
        quantum=quantum,
        failfast=failfast,
        return_eigensolution=False,
    )

    # Assign new positions (with displacements) and velocities
    atoms.positions += d_ac
    atoms.set_velocities(v_ac)
