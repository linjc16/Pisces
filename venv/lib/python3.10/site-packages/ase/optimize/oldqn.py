# fmt: off

# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
Quasi-Newton algorithm
"""

__docformat__ = 'reStructuredText'

import time
from typing import IO, Optional, Union

import numpy as np
from numpy.linalg import eigh

from ase import Atoms
from ase.optimize.optimize import Optimizer


def f(lamda, Gbar, b, radius):
    b1 = b - lamda
    b1[abs(b1) < 1e-40] = 1e-40  # avoid divide-by-zero
    gbar_b_lamda = Gbar / b1  # only compute once
    g = radius**2 - np.dot(gbar_b_lamda, gbar_b_lamda)
    return g


def scale_radius_energy(f, r):
    scale = 1.0
#       if(r<=0.01):
#               return scale

    if f < 0.01:
        scale *= 1.4
    if f < 0.05:
        scale *= 1.4
    if f < 0.10:
        scale *= 1.4
    if f < 0.40:
        scale *= 1.4

    if f > 0.5:
        scale *= 1. / 1.4
    if f > 0.7:
        scale *= 1. / 1.4
    if f > 1.0:
        scale *= 1. / 1.4

    return scale


def scale_radius_force(f, r):
    scale = 1.0
#       if(r<=0.01):
#               return scale
    g = abs(f - 1)
    if g < 0.01:
        scale *= 1.4
    if g < 0.05:
        scale *= 1.4
    if g < 0.10:
        scale *= 1.4
    if g < 0.40:
        scale *= 1.4

    if g > 0.5:
        scale *= 1. / 1.4
    if g > 0.7:
        scale *= 1. / 1.4
    if g > 1.0:
        scale *= 1. / 1.4

    return scale


def find_lamda(upperlimit, Gbar, b, radius):
    lowerlimit = upperlimit
    step = 0.1
    i = 0

    while f(lowerlimit, Gbar, b, radius) < 0:
        lowerlimit -= step
        i += 1

        # if many iterations are required, dynamically scale step for efficiency
        if i % 100 == 0:
            step *= 1.25

    converged = False

    while not converged:

        midt = (upperlimit + lowerlimit) / 2.
        lamda = midt
        fmidt = f(midt, Gbar, b, radius)
        fupper = f(upperlimit, Gbar, b, radius)

        if fupper * fmidt < 0:
            lowerlimit = midt
        else:
            upperlimit = midt

        if abs(upperlimit - lowerlimit) < 1e-6:
            converged = True

    return lamda


class GoodOldQuasiNewton(Optimizer):

    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = '-',
        trajectory: Optional[str] = None,
        fmax=None,
        converged=None,
        hessianupdate: str = 'BFGS',
        hessian=None,
        forcemin: bool = True,
        verbosity: bool = False,
        maxradius: Optional[float] = None,
        diagonal: float = 20.0,
        radius: Optional[float] = None,
        transitionstate: bool = False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        atoms: :class:`~ase.Atoms`
            The Atoms object to relax.

        restart: str
            File used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: str
            File used to store trajectory of atomic movement.

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

        Optimizer.__init__(self, atoms, restart, logfile, trajectory, **kwargs)

        self.eps = 1e-12
        self.hessianupdate = hessianupdate
        self.forcemin = forcemin
        self.verbosity = verbosity
        self.diagonal = diagonal

        n = len(self.optimizable) * 3
        if radius is None:
            self.radius = 0.05 * np.sqrt(n) / 10.0
        else:
            self.radius = radius

        if maxradius is None:
            self.maxradius = 0.5 * np.sqrt(n)
        else:
            self.maxradius = maxradius

        # 0.01 < radius < maxradius
        self.radius = max(min(self.radius, self.maxradius), 0.0001)

        self.transitionstate = transitionstate
        self.t0 = time.time()

    def initialize(self):
        pass

    def write_log(self, text):
        if self.logfile is not None:
            self.logfile.write(text + '\n')
            self.logfile.flush()

    def set_hessian(self, hessian):
        self.hessian = hessian

    def get_hessian(self):
        if not hasattr(self, 'hessian'):
            self.set_default_hessian()
        return self.hessian

    def set_default_hessian(self):
        # set unit matrix
        n = len(self.optimizable) * 3
        hessian = np.zeros((n, n))
        for i in range(n):
            hessian[i][i] = self.diagonal
        self.set_hessian(hessian)

    def update_hessian(self, pos, G):
        import copy
        if hasattr(self, 'oldG'):
            if self.hessianupdate == 'BFGS':
                self.update_hessian_bfgs(pos, G)
            elif self.hessianupdate == 'Powell':
                self.update_hessian_powell(pos, G)
            else:
                self.update_hessian_bofill(pos, G)
        else:
            if not hasattr(self, 'hessian'):
                self.set_default_hessian()

        self.oldpos = copy.copy(pos)
        self.oldG = copy.copy(G)

        if self.verbosity:
            print('hessian ', self.hessian)

    def update_hessian_bfgs(self, pos, G):
        n = len(self.hessian)
        dgrad = G - self.oldG
        dpos = pos - self.oldpos
        dotg = np.dot(dgrad, dpos)
        tvec = np.dot(dpos, self.hessian)
        dott = np.dot(dpos, tvec)
        if (abs(dott) > self.eps) and (abs(dotg) > self.eps):
            for i in range(n):
                for j in range(n):
                    h = dgrad[i] * dgrad[j] / dotg - tvec[i] * tvec[j] / dott
                    self.hessian[i][j] += h

    def update_hessian_powell(self, pos, G):
        n = len(self.hessian)
        dgrad = G - self.oldG
        dpos = pos - self.oldpos
        absdpos = np.dot(dpos, dpos)
        if absdpos < self.eps:
            return

        dotg = np.dot(dgrad, dpos)
        tvec = dgrad - np.dot(dpos, self.hessian)
        tvecdpos = np.dot(tvec, dpos)
        ddot = tvecdpos / absdpos

        dott = np.dot(dpos, tvec)
        if (abs(dott) > self.eps) and (abs(dotg) > self.eps):
            for i in range(n):
                for j in range(n):
                    h = tvec[i] * dpos[j] + dpos[i] * \
                        tvec[j] - ddot * dpos[i] * dpos[j]
                    h *= 1. / absdpos
                    self.hessian[i][j] += h

    def update_hessian_bofill(self, pos, G):
        print('update Bofill')
        n = len(self.hessian)
        dgrad = G - self.oldG
        dpos = pos - self.oldpos
        absdpos = np.dot(dpos, dpos)
        if absdpos < self.eps:
            return
        dotg = np.dot(dgrad, dpos)
        tvec = dgrad - np.dot(dpos, self.hessian)
        tvecdot = np.dot(tvec, tvec)
        tvecdpos = np.dot(tvec, dpos)

        coef1 = 1. - tvecdpos * tvecdpos / (absdpos * tvecdot)
        coef2 = (1. - coef1) * absdpos / tvecdpos
        coef3 = coef1 * tvecdpos / absdpos

        dott = np.dot(dpos, tvec)
        if (abs(dott) > self.eps) and (abs(dotg) > self.eps):
            for i in range(n):
                for j in range(n):
                    h = coef1 * (tvec[i] * dpos[j] + dpos[i] * tvec[j]) - \
                        dpos[i] * dpos[j] * coef3 + coef2 * tvec[i] * tvec[j]
                    h *= 1. / absdpos
                    self.hessian[i][j] += h

    def step(self, forces=None):
        """ Do one QN step
        """

        if forces is None:
            forces = self.optimizable.get_forces()

        pos = self.optimizable.get_positions().ravel()
        G = -self.optimizable.get_forces().ravel()

        energy = self.optimizable.get_potential_energy()

        if hasattr(self, 'oldenergy'):
            self.write_log('energies ' + str(energy) +
                           ' ' + str(self.oldenergy))

            if self.forcemin:
                de = 1e-4
            else:
                de = 1e-2

            if self.transitionstate:
                de = 0.2

            if (energy - self.oldenergy) > de:
                self.write_log('reject step')
                self.optimizable.set_positions(self.oldpos.reshape((-1, 3)))
                G = self.oldG
                energy = self.oldenergy
                self.radius *= 0.5
            else:
                self.update_hessian(pos, G)
                de = energy - self.oldenergy
                forces = 1.0
                if self.forcemin:
                    self.write_log(
                        "energy change; actual: %f estimated: %f " %
                        (de, self.energy_estimate))
                    if abs(self.energy_estimate) > self.eps:
                        forces = abs((de / self.energy_estimate) - 1)
                        self.write_log('Energy prediction factor '
                                       + str(forces))
                        # fg = self.get_force_prediction(G)
                        self.radius *= scale_radius_energy(forces, self.radius)

                else:
                    self.write_log("energy change; actual: %f " % (de))
                    self.radius *= 1.5

                fg = self.get_force_prediction(G)
                self.write_log("Scale factors %f %f " %
                               (scale_radius_energy(forces, self.radius),
                                scale_radius_force(fg, self.radius)))

            self.radius = max(min(self.radius, self.maxradius), 0.0001)
        else:
            self.update_hessian(pos, G)

        self.write_log("new radius %f " % (self.radius))
        self.oldenergy = energy

        b, V = eigh(self.hessian)
        V = V.T.copy()
        self.V = V

        # calculate projection of G onto eigenvectors V
        Gbar = np.dot(G, np.transpose(V))

        lamdas = self.get_lambdas(b, Gbar)

        D = -Gbar / (b - lamdas)
        n = len(D)
        step = np.zeros(n)
        for i in range(n):
            step += D[i] * V[i]

        pos = self.optimizable.get_positions().ravel()
        pos += step

        energy_estimate = self.get_energy_estimate(D, Gbar, b)
        self.energy_estimate = energy_estimate
        self.gbar_estimate = self.get_gbar_estimate(D, Gbar, b)
        self.old_gbar = Gbar

        self.optimizable.set_positions(pos.reshape((-1, 3)))

    def get_energy_estimate(self, D, Gbar, b):

        de = 0.0
        for n in range(len(D)):
            de += D[n] * Gbar[n] + 0.5 * D[n] * b[n] * D[n]
        return de

    def get_gbar_estimate(self, D, Gbar, b):
        gbar_est = (D * b) + Gbar
        self.write_log('Abs Gbar estimate ' + str(np.dot(gbar_est, gbar_est)))
        return gbar_est

    def get_lambdas(self, b, Gbar):
        lamdas = np.zeros(len(b))

        D = -Gbar / b
        absD = np.sqrt(np.dot(D, D))

        eps = 1e-12
        nminus = self.get_hessian_inertia(b)

        if absD < self.radius:
            if not self.transitionstate:
                self.write_log('Newton step')
                return lamdas
            else:
                if nminus == 1:
                    self.write_log('Newton step')
                    return lamdas
                else:
                    self.write_log(
                        "Wrong inertia of Hessian matrix: %2.2f %2.2f " %
                        (b[0], b[1]))

        else:
            self.write_log("Corrected Newton step: abs(D) = %2.2f " % (absD))

        if not self.transitionstate:
            # upper limit
            upperlimit = min(0, b[0]) - eps
            lamda = find_lamda(upperlimit, Gbar, b, self.radius)
            lamdas += lamda
        else:
            # upperlimit
            upperlimit = min(-b[0], b[1], 0) - eps
            lamda = find_lamda(upperlimit, Gbar, b, self.radius)
            lamdas += lamda
            lamdas[0] -= 2 * lamda

        return lamdas

    def get_hessian_inertia(self, eigenvalues):
        # return number of negative modes
        self.write_log("eigenvalues {:2.2f} {:2.2f} {:2.2f} ".format(
            eigenvalues[0], eigenvalues[1], eigenvalues[2]))
        n = 0
        while eigenvalues[n] < 0:
            n += 1
        return n

    def get_force_prediction(self, G):
        # return measure of how well the forces are predicted
        Gbar = np.dot(G, np.transpose(self.V))
        dGbar_actual = Gbar - self.old_gbar
        dGbar_predicted = Gbar - self.gbar_estimate

        f = np.dot(dGbar_actual, dGbar_predicted) / \
            np.dot(dGbar_actual, dGbar_actual)
        self.write_log('Force prediction factor ' + str(f))
        return f
