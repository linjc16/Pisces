# fmt: off

from typing import IO, Any, Callable, Dict, List, Optional, Union

import numpy as np

from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.utils import deprecated


def _forbid_maxmove(args: List, kwargs: Dict[str, Any]) -> bool:
    """Set maxstep with maxmove if not set."""
    maxstep_index = 6
    maxmove_index = 7

    def _pop_arg(name: str) -> Any:
        to_pop = None
        if len(args) > maxmove_index:
            to_pop = args[maxmove_index]
            args[maxmove_index] = None

        elif name in kwargs:
            to_pop = kwargs[name]
            del kwargs[name]
        return to_pop

    if len(args) > maxstep_index and args[maxstep_index] is None:
        value = args[maxstep_index] = _pop_arg("maxmove")
    elif kwargs.get("maxstep", None) is None:
        value = kwargs["maxstep"] = _pop_arg("maxmove")
    else:
        return False

    return value is not None


class FIRE(Optimizer):
    @deprecated(
        "Use of `maxmove` is deprecated. Use `maxstep` instead.",
        category=FutureWarning,
        callback=_forbid_maxmove,
    )
    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = '-',
        trajectory: Optional[str] = None,
        dt: float = 0.1,
        maxstep: Optional[float] = None,
        maxmove: Optional[float] = None,
        dtmax: float = 1.0,
        Nmin: int = 5,
        finc: float = 1.1,
        fdec: float = 0.5,
        astart: float = 0.1,
        fa: float = 0.99,
        a: float = 0.1,
        downhill_check: bool = False,
        position_reset_callback: Optional[Callable] = None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        atoms: :class:`~ase.Atoms`
            The Atoms object to relax.

        restart: str
            JSON file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: str
            Trajectory file used to store optimisation path.

        dt: float
            Initial time step. Defualt value is 0.1

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2).

        dtmax: float
            Maximum time step. Default value is 1.0

        Nmin: int
            Number of steps to wait after the last time the dot product of
            the velocity and force is negative (P in The FIRE article) before
            increasing the time step. Default value is 5.

        finc: float
            Factor to increase the time step. Default value is 1.1

        fdec: float
            Factor to decrease the time step. Default value is 0.5

        astart: float
            Initial value of the parameter a. a is the Coefficient for
            mixing the velocity and the force. Called alpha in the FIRE article.
            Default value 0.1.

        fa: float
            Factor to decrease the parameter alpha. Default value is 0.99

        a: float
            Coefficient for mixing the velocity and the force. Called
            alpha in the FIRE article. Default value 0.1.

        downhill_check: bool
            Downhill check directly compares potential energies of subsequent
            steps of the FIRE algorithm rather than relying on the current
            product v*f that is positive if the FIRE dynamics moves downhill.
            This can detect numerical issues where at large time steps the step
            is uphill in energy even though locally v*f is positive, i.e. the
            algorithm jumps over a valley because of a too large time step.

        position_reset_callback: function(atoms, r, e, e_last)
            Function that takes current *atoms* object, an array of position
            *r* that the optimizer will revert to, current energy *e* and
            energy of last step *e_last*. This is only called if e > e_last.

        kwargs : dict, optional
            Extra arguments passed to
            :class:`~ase.optimize.optimize.Optimizer`.

        .. deprecated:: 3.19.3
            Use of ``maxmove`` is deprecated; please use ``maxstep``.

        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, **kwargs)

        self.dt = dt

        self.Nsteps = 0

        if maxstep is not None:
            self.maxstep = maxstep
        else:
            self.maxstep = self.defaults["maxstep"]

        self.dtmax = dtmax
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa
        self.a = a
        self.downhill_check = downhill_check
        self.position_reset_callback = position_reset_callback

    def initialize(self):
        self.v = None

    def read(self):
        self.v, self.dt = self.load()

    def step(self, f=None):
        optimizable = self.optimizable

        if f is None:
            f = optimizable.get_forces()

        if self.v is None:
            self.v = np.zeros((len(optimizable), 3))
            if self.downhill_check:
                self.e_last = optimizable.get_potential_energy()
                self.r_last = optimizable.get_positions().copy()
                self.v_last = self.v.copy()
        else:
            is_uphill = False
            if self.downhill_check:
                e = optimizable.get_potential_energy()
                # Check if the energy actually decreased
                if e > self.e_last:
                    # If not, reset to old positions...
                    if self.position_reset_callback is not None:
                        self.position_reset_callback(
                            optimizable, self.r_last, e,
                            self.e_last)
                    optimizable.set_positions(self.r_last)
                    is_uphill = True
                self.e_last = optimizable.get_potential_energy()
                self.r_last = optimizable.get_positions().copy()
                self.v_last = self.v.copy()

            vf = np.vdot(f, self.v)
            if vf > 0.0 and not is_uphill:
                self.v = (1.0 - self.a) * self.v + self.a * f / np.sqrt(
                    np.vdot(f, f)) * np.sqrt(np.vdot(self.v, self.v))
                if self.Nsteps > self.Nmin:
                    self.dt = min(self.dt * self.finc, self.dtmax)
                    self.a *= self.fa
                self.Nsteps += 1
            else:
                self.v[:] *= 0.0
                self.a = self.astart
                self.dt *= self.fdec
                self.Nsteps = 0

        self.v += self.dt * f
        dr = self.dt * self.v
        normdr = np.sqrt(np.vdot(dr, dr))
        if normdr > self.maxstep:
            dr = self.maxstep * dr / normdr
        r = optimizable.get_positions()
        optimizable.set_positions(r + dr)
        self.dump((self.v, self.dt))
