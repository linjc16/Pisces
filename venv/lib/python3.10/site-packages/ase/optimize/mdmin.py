from typing import IO, Optional, Union

import numpy as np

from ase import Atoms
from ase.optimize.optimize import Optimizer


class MDMin(Optimizer):
    # default parameters
    defaults = {**Optimizer.defaults, 'dt': 0.2}

    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = '-',
        trajectory: Optional[str] = None,
        dt: Optional[float] = None,
        maxstep: Optional[float] = None,
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

        trajectory: str
            Trajectory file used to store optimisation path.

        logfile: str
            Text file used to write summary information.

        dt: float
            Time step for integrating the equation of motion.

        maxstep: float
            Spatial step limit in Angstrom. This allows larger values of dt
            while being more robust to instabilities in the optimization.

        kwargs : dict, optional
            Extra arguments passed to
            :class:`~ase.optimize.optimize.Optimizer`.

        """
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, **kwargs)

        self.dt = dt or self.defaults['dt']
        self.maxstep = maxstep or self.defaults['maxstep']

    def initialize(self):
        self.v = None

    def read(self):
        self.v, self.dt = self.load()

    def step(self, forces=None):
        optimizable = self.optimizable

        if forces is None:
            forces = optimizable.get_forces()

        if self.v is None:
            self.v = np.zeros((len(optimizable), 3))
        else:
            self.v += 0.5 * self.dt * forces
            # Correct velocities:
            vf = np.vdot(self.v, forces)
            if vf < 0.0:
                self.v[:] = 0.0
            else:
                self.v[:] = forces * vf / np.vdot(forces, forces)

        self.v += 0.5 * self.dt * forces
        pos = optimizable.get_positions()
        dpos = self.dt * self.v

        # For any dpos magnitude larger than maxstep, scaling
        # is <1. We add a small float to prevent overflows/zero-div errors.
        # All displacement vectors (rows) of dpos which have a norm larger
        # than self.maxstep are scaled to it.
        scaling = self.maxstep / (1e-6 + np.max(np.linalg.norm(dpos, axis=1)))
        dpos *= np.clip(scaling, 0.0, 1.0)
        optimizable.set_positions(pos + dpos)
        self.dump((self.v, self.dt))
