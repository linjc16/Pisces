# fmt: off

# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************

import time
from typing import IO, Optional, Union

import numpy as np
from numpy import absolute, eye, isinf, sqrt

from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.utils.linesearch import LineSearch

# These have been copied from Numeric's MLab.py
# I don't think they made the transition to scipy_core

# Modified from scipy_optimize
abs = absolute
pymin = min
pymax = max
__version__ = '0.1'


class BFGSLineSearch(Optimizer):
    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = '-',
        maxstep: float = None,
        trajectory: Optional[str] = None,
        c1: float = 0.23,
        c2: float = 0.46,
        alpha: float = 10.0,
        stpmax: float = 50.0,
        **kwargs,
    ):
        """Optimize atomic positions in the BFGSLineSearch algorithm, which
        uses both forces and potential energy information.

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

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Angstroms).

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        kwargs : dict, optional
            Extra arguments passed to
            :class:`~ase.optimize.optimize.Optimizer`.

        """
        if maxstep is None:
            self.maxstep = self.defaults['maxstep']
        else:
            self.maxstep = maxstep
        self.stpmax = stpmax
        self.alpha = alpha
        self.H = None
        self.c1 = c1
        self.c2 = c2
        self.force_calls = 0
        self.function_calls = 0
        self.r0 = None
        self.g0 = None
        self.e0 = None
        self.load_restart = False
        self.task = 'START'
        self.rep_count = 0
        self.p = None
        self.alpha_k = None
        self.no_update = False
        self.replay = False

        Optimizer.__init__(self, atoms, restart, logfile, trajectory, **kwargs)

    def read(self):
        self.r0, self.g0, self.e0, self.task, self.H = self.load()
        self.load_restart = True

    def reset(self):
        self.H = None
        self.r0 = None
        self.g0 = None
        self.e0 = None
        self.rep_count = 0

    def step(self, forces=None):
        optimizable = self.optimizable

        if forces is None:
            forces = optimizable.get_forces()

        r = optimizable.get_positions()
        r = r.reshape(-1)
        g = -forces.reshape(-1) / self.alpha
        p0 = self.p
        self.update(r, g, self.r0, self.g0, p0)
        # o,v = np.linalg.eigh(self.B)
        e = self.func(r)

        self.p = -np.dot(self.H, g)
        p_size = np.sqrt((self.p**2).sum())
        if p_size <= np.sqrt(len(optimizable) * 1e-10):
            self.p /= (p_size / np.sqrt(len(optimizable) * 1e-10))
        ls = LineSearch()
        self.alpha_k, e, self.e0, self.no_update = \
            ls._line_search(self.func, self.fprime, r, self.p, g, e, self.e0,
                            maxstep=self.maxstep, c1=self.c1,
                            c2=self.c2, stpmax=self.stpmax)
        if self.alpha_k is None:
            raise RuntimeError("LineSearch failed!")

        dr = self.alpha_k * self.p
        optimizable.set_positions((r + dr).reshape(len(optimizable), -1))
        self.r0 = r
        self.g0 = g
        self.dump((self.r0, self.g0, self.e0, self.task, self.H))

    def update(self, r, g, r0, g0, p0):
        self.I = eye(len(self.optimizable) * 3, dtype=int)
        if self.H is None:
            self.H = eye(3 * len(self.optimizable))
            # self.B = np.linalg.inv(self.H)
            return
        else:
            dr = r - r0
            dg = g - g0
            # self.alpha_k can be None!!!
            if not (((self.alpha_k or 0) > 0 and
                    abs(np.dot(g, p0)) - abs(np.dot(g0, p0)) < 0) or
                    self.replay):
                return
            if self.no_update is True:
                print('skip update')
                return

            try:  # this was handled in numeric, let it remain for more safety
                rhok = 1.0 / (np.dot(dg, dr))
            except ZeroDivisionError:
                rhok = 1000.0
                print("Divide-by-zero encountered: rhok assumed large")
            if isinf(rhok):  # this is patch for np
                rhok = 1000.0
                print("Divide-by-zero encountered: rhok assumed large")
            A1 = self.I - dr[:, np.newaxis] * dg[np.newaxis, :] * rhok
            A2 = self.I - dg[:, np.newaxis] * dr[np.newaxis, :] * rhok
            self.H = (np.dot(A1, np.dot(self.H, A2)) +
                      rhok * dr[:, np.newaxis] * dr[np.newaxis, :])
            # self.B = np.linalg.inv(self.H)

    def func(self, x):
        """Objective function for use of the optimizers"""
        self.optimizable.set_positions(x.reshape(-1, 3))
        self.function_calls += 1
        # Scale the problem as SciPy uses I as initial Hessian.
        return self.optimizable.get_potential_energy() / self.alpha

    def fprime(self, x):
        """Gradient of the objective function for use of the optimizers"""
        self.optimizable.set_positions(x.reshape(-1, 3))
        self.force_calls += 1
        # Remember that forces are minus the gradient!
        # Scale the problem as SciPy uses I as initial Hessian.
        forces = self.optimizable.get_forces().reshape(-1)
        return - forces / self.alpha

    def replay_trajectory(self, traj):
        """Initialize hessian from old trajectory."""
        self.replay = True
        from ase.utils import IOContext

        with IOContext() as files:
            if isinstance(traj, str):
                from ase.io.trajectory import Trajectory
                traj = files.closelater(Trajectory(traj, mode='r'))

            r0 = None
            g0 = None
            for i in range(len(traj) - 1):
                r = traj[i].get_positions().ravel()
                g = - traj[i].get_forces().ravel() / self.alpha
                self.update(r, g, r0, g0, self.p)
                self.p = -np.dot(self.H, g)
                r0 = r.copy()
                g0 = g.copy()
            self.r0 = r0
            self.g0 = g0

    def log(self, forces=None):
        if self.logfile is None:
            return
        if forces is None:
            forces = self.optimizable.get_forces()
        fmax = sqrt((forces**2).sum(axis=1).max())
        e = self.optimizable.get_potential_energy()
        T = time.localtime()
        name = self.__class__.__name__
        w = self.logfile.write
        if self.nsteps == 0:
            w('%s  %4s[%3s] %8s %15s  %12s\n' %
              (' ' * len(name), 'Step', 'FC', 'Time', 'Energy', 'fmax'))
        w('%s:  %3d[%3d] %02d:%02d:%02d %15.6f %12.4f\n'
            % (name, self.nsteps, self.force_calls, T[3], T[4], T[5], e,
               fmax))
        self.logfile.flush()


def wrap_function(function, args):
    ncalls = [0]

    def function_wrapper(x):
        ncalls[0] += 1
        return function(x, *args)
    return ncalls, function_wrapper
