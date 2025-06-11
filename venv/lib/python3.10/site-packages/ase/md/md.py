# fmt: off

"""Molecular Dynamics."""
import warnings
from typing import IO, Optional, Union

import numpy as np

from ase import Atoms, units
from ase.md.logger import MDLogger
from ase.optimize.optimize import Dynamics


def process_temperature(
    temperature: Optional[float],
    temperature_K: Optional[float],
    orig_unit: str,
) -> float:
    """Handle that temperature can be specified in multiple units.

    For at least a transition period, molecular dynamics in ASE can
    have the temperature specified in either Kelvin or Electron
    Volt.  The different MD algorithms had different defaults, by
    forcing the user to explicitly choose a unit we can resolve
    this.  Using the original method then will issue a
    FutureWarning.

    Four parameters:

    temperature: None or float
        The original temperature specification in whatever unit was
        historically used.  A warning is issued if this is not None and
        the historical unit was eV.

    temperature_K: None or float
        Temperature in Kelvin.

    orig_unit: str
        Unit used for the `temperature`` parameter.  Must be 'K' or 'eV'.

    Exactly one of the two temperature parameters must be different from
    None, otherwise an error is issued.

    Return value: Temperature in Kelvin.
    """
    if (temperature is not None) + (temperature_K is not None) != 1:
        raise TypeError("Exactly one of the parameters 'temperature',"
                        + " and 'temperature_K', must be given")
    if temperature is not None:
        w = "Specify the temperature in K using the 'temperature_K' argument"
        if orig_unit == 'K':
            return temperature
        elif orig_unit == 'eV':
            warnings.warn(FutureWarning(w))
            return temperature / units.kB
        else:
            raise ValueError("Unknown temperature unit " + orig_unit)

    assert temperature_K is not None
    return temperature_K


class MolecularDynamics(Dynamics):
    """Base-class for all MD classes."""

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        trajectory: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        loginterval: int = 1,
        **kwargs,
    ):
        """Molecular Dynamics object.

        Parameters
        ----------
        atoms : Atoms object
            The Atoms object to operate on.

        timestep : float
            The time step in ASE time units.

        trajectory : Trajectory object or str
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        logfile : file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        loginterval : int, default: 1
            Only write a log line for every *loginterval* time steps.

        kwargs : dict, optional
            Extra arguments passed to :class:`~ase.optimize.optimize.Dynamics`.
        """
        # dt as to be attached _before_ parent class is initialized
        self.dt = timestep

        super().__init__(
            atoms,
            logfile=None,
            trajectory=trajectory,
            loginterval=loginterval,
            **kwargs,
        )

        # Some codes (e.g. Asap) may be using filters to
        # constrain atoms or do other things.  Current state of the art
        # is that "atoms" must be either Atoms or Filter in order to
        # work with dynamics.
        #
        # In the future, we should either use a special role interface
        # for MD, or we should ensure that the input is *always* a Filter.
        # That way we won't need to test multiple cases.  Currently,
        # we do not test /any/ kind of MD with any kind of Filter in ASE.
        self.atoms = atoms
        self.masses = self.atoms.get_masses()

        if 0 in self.masses:
            warnings.warn('Zero mass encountered in atoms; this will '
                          'likely lead to errors if the massless atoms '
                          'are unconstrained.')

        self.masses.shape = (-1, 1)

        if not self.atoms.has('momenta'):
            self.atoms.set_momenta(np.zeros([len(self.atoms), 3]))

        if logfile:
            logger = self.closelater(
                MDLogger(dyn=self, atoms=atoms, logfile=logfile))
            self.attach(logger, loginterval)

    def todict(self):
        return {'type': 'molecular-dynamics',
                'md-type': self.__class__.__name__,
                'timestep': self.dt}

    def irun(self, steps=50):
        """Run molecular dynamics algorithm as a generator.

        Parameters
        ----------
        steps : int, default=DEFAULT_MAX_STEPS
            Number of molecular dynamics steps to be run.

        Yields
        ------
        converged : bool
            True if the maximum number of steps are reached.
        """
        return Dynamics.irun(self, steps=steps)

    def run(self, steps=50):
        """Run molecular dynamics algorithm.

        Parameters
        ----------
        steps : int, default=DEFAULT_MAX_STEPS
            Number of molecular dynamics steps to be run.

        Returns
        -------
        converged : bool
            True if the maximum number of steps are reached.
        """
        return Dynamics.run(self, steps=steps)

    def get_time(self):
        return self.nsteps * self.dt

    def converged(self):
        """ MD is 'converged' when number of maximum steps is reached. """
        return self.nsteps >= self.max_steps

    def _get_com_velocity(self, velocity):
        """Return the center of mass velocity.
        Internal use only. This function can be reimplemented by Asap.
        """
        return np.dot(self.masses.ravel(), velocity) / self.masses.sum()

    # Make the process_temperature function available to subclasses
    # as a static method.  This makes it easy for MD objects to use
    # it, while functions in md.velocitydistribution have access to it
    # as a function.
    _process_temperature = staticmethod(process_temperature)
