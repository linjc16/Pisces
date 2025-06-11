# fmt: off

from typing import IO, Optional, Union

import numpy as np

from ase import Atoms
from ase.optimize.optimize import Dynamics


def subtract_projection(a, b):
    '''returns new vector that removes vector a's projection vector b. Is
    also equivalent to the vector rejection.'''
    aout = a - np.vdot(a, b) / np.vdot(b, b) * b
    return aout


def normalize(a):
    '''Makes a unit vector out of a vector'''
    return a / np.linalg.norm(a)


class ContourExploration(Dynamics):

    def __init__(
        self,
        atoms: Atoms,
        maxstep: float = 0.5,
        parallel_drift: float = 0.1,
        energy_target: Optional[float] = None,
        angle_limit: Optional[float] = 20.0,
        potentiostat_step_scale: Optional[float] = None,
        remove_translation: bool = False,
        use_frenet_serret: bool = True,
        initialization_step_scale: float = 1e-2,
        use_target_shift: bool = True,
        target_shift_previous_steps: int = 10,
        use_tangent_curvature: bool = False,
        rng=np.random,
        force_consistent: Optional[bool] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        append_trajectory: bool = False,
        loginterval: int = 1,
    ):
        """Contour Exploration object.

        Parameters:

        atoms: Atoms object
            The Atoms object to operate on. Atomic velocities are required for
            the method. If the atoms object does not contain velocities,
            random ones will be applied.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.5 Å).

        parallel_drift: float
            The fraction of the update step that is parallel to the contour but
            in a random direction. Used to break symmetries.

        energy_target: float
            The total system potential energy for that the potentiostat attepts
            to maintain. (defaults the initial potential energy)

        angle_limit: float or None
            Limits the stepsize to a maximum change of direction angle using the
            curvature. Gives a scale-free means of tuning the stepsize on the
            fly. Typically less than 30 degrees gives reasonable results but
            lower angle limits result in higher potentiostatic accuracy. Units
            of degrees. (default 20°)

        potentiostat_step_scale: float or None
            Scales the size of the potentiostat step. The potentiostat step is
            determined by linear extrapolation from the current potential energy
            to the target_energy with the current forces. A
            potentiostat_step_scale > 1.0 overcorrects and < 1.0
            undercorrects. By default, a simple heuristic is used to selected
            the valued based on the parallel_drift. (default None)

        remove_translation: boolean
            When True, the net momentum is removed at each step. Improves
            potentiostatic accuracy slightly for bulk systems but should not be
            used with constraints. (default False)

        use_frenet_serret: Bool
            Controls whether or not the Taylor expansion of the Frenet-Serret
            formulas for curved path extrapolation are used. Required for using
            angle_limit based step scalling. (default True)

        initialization_step_scale: float
            Controls the scale of the initial step as a multiple of maxstep.
            (default 1e-2)

        use_target_shift: boolean
            Enables shifting of the potentiostat target to compensate for
            systematic undercorrection or overcorrection by the potentiostat.
            Uses the average of the *target_shift_previous_steps* to prevent
            coupled occilations. (default True)

        target_shift_previous_steps: int
            The number of pevious steps to average when using use_target_shift.
            (default 10)

        use_tangent_curvature: boolean
            Use the velocity unit tangent rather than the contour normals from
            forces to compute the curvature. Usually not as accurate.
            (default False)

        rng: a random number generator
            Lets users control the random number generator for the
            parallel_drift vector. (default numpy.random)

         force_consistent: boolean
             (default None)

        trajectory: Trajectory object or str  (optional)
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Default: None.

        logfile: file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.  Default: None.

        loginterval: int (optional)
            Only write a log line for every *loginterval* time steps.
            Default: 1

        append_trajectory: boolean
            Defaults to False, which causes the trajectory file to be
            overwriten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.
        """

        if potentiostat_step_scale is None:
            # a heuristic guess since most systems will overshoot when there is
            # drift
            self.potentiostat_step_scale = 1.1 + 0.6 * parallel_drift
        else:
            self.potentiostat_step_scale = potentiostat_step_scale

        self.rng = rng
        self.remove_translation = remove_translation
        self.use_frenet_serret = use_frenet_serret
        self.use_tangent_curvature = use_tangent_curvature
        self.initialization_step_scale = initialization_step_scale
        self.maxstep = maxstep
        self.angle_limit = angle_limit
        self.parallel_drift = parallel_drift
        self.use_target_shift = use_target_shift

        # These will be populated once self.step() is called, but could be set
        # after instantiating with ce = ContourExploration(...) like so:
        # ce.Nold = Nold
        # ce.r_old = atoms_old.get_positions()
        # ce.Told  = Told
        # to resume a previous contour trajectory.

        self.T = None
        self.Told = None
        self.N = None
        self.Nold = None
        self.r_old = None
        self.r = None

        if energy_target is None:
            self.energy_target = atoms.get_potential_energy(
                force_consistent=True)
        else:
            self.energy_target = energy_target

        # Initizing the previous steps at the target energy slows
        # target_shifting untill the system has had
        # 'target_shift_previous_steps' steps to equilibrate and should prevent
        # occilations. These need to be initialized before the initialize_old
        # step to prevent a crash
        self.previous_energies = np.full(target_shift_previous_steps,
                                         self.energy_target)

        # these first two are purely for logging,
        # auto scaling will still occur
        # and curvature will still be found if use_frenet_serret == True
        self.step_size = 0.0
        self.curvature = 0

        # loginterval exists for the MolecularDynamics class but not for
        # the more general Dynamics class
        Dynamics.__init__(self, atoms,
                          logfile, trajectory,  # loginterval,
                          append_trajectory=append_trajectory,
                          )

        self._actual_atoms = atoms

        # we need velocities or NaNs will be produced,
        # if none are provided we make random ones
        velocities = self._actual_atoms.get_velocities()
        if np.linalg.norm(velocities) < 1e-6:
            # we have to pass dimension since atoms are not yet stored
            atoms.set_velocities(self.rand_vect())

    # Required stuff for Dynamics
    def todict(self):
        return {'type': 'contour-exploration',
                'dyn-type': self.__class__.__name__,
                'stepsize': self.step_size}

    def run(self, steps=50):
        """ Call Dynamics.run and adjust max_steps """
        return Dynamics.run(self, steps=steps)

    def log(self):
        if self.logfile is not None:
            # name = self.__class__.__name__
            if self.nsteps == 0:
                args = (
                    "Step",
                    "Energy_Target",
                    "Energy",
                    "Curvature",
                    "Step_Size",
                    "Energy_Deviation_per_atom")
                msg = "# %4s %15s %15s %12s %12s %15s\n" % args
                self.logfile.write(msg)
            e = self._actual_atoms.get_potential_energy(force_consistent=True)
            dev_per_atom = (e - self.energy_target) / len(self._actual_atoms)
            args = (
                self.nsteps,
                self.energy_target,
                e,
                self.curvature,
                self.step_size,
                dev_per_atom)
            msg = "%6d %15.6f %15.6f %12.6f %12.6f %24.9f\n" % args
            self.logfile.write(msg)

            self.logfile.flush()

    def rand_vect(self):
        '''Returns a random (Natoms,3) vector'''
        vect = self.rng.normal(size=(len(self._actual_atoms), 3))
        return vect

    def create_drift_unit_vector(self, N, T):
        '''Creates a random drift unit vector with no projection on N or T and
        with out a net translation so systems don't wander'''
        drift = self.rand_vect()
        drift = subtract_projection(drift, N)
        drift = subtract_projection(drift, T)
        # removes net translation, so systems don't wander
        drift = drift - drift.sum(axis=0) / len(self._actual_atoms)
        D = normalize(drift)
        return D

    def compute_step_contributions(self, potentiostat_step_size):
        '''Computes the orthogonal component sizes of the step so that the net
        step obeys the smaller of step_size or maxstep.'''
        if abs(potentiostat_step_size) < self.step_size:
            delta_s_perpendicular = potentiostat_step_size
            contour_step_size = np.sqrt(
                self.step_size**2 - potentiostat_step_size**2)
            delta_s_parallel = np.sqrt(
                1 - self.parallel_drift**2) * contour_step_size
            delta_s_drift = contour_step_size * self.parallel_drift

        else:
            # in this case all priority goes to potentiostat terms
            delta_s_parallel = 0.0
            delta_s_drift = 0.0
            delta_s_perpendicular = np.sign(
                potentiostat_step_size) * self.step_size

        return delta_s_perpendicular, delta_s_parallel, delta_s_drift

    def _compute_update_without_fs(self, potentiostat_step_size, scale=1.0):
        '''Only uses the forces to compute an orthogonal update vector'''

        # Without the use of curvature there is no way to estimate the
        # limiting step size
        self.step_size = self.maxstep * scale

        delta_s_perpendicular, delta_s_parallel, delta_s_drift = \
            self.compute_step_contributions(
                potentiostat_step_size)

        dr_perpendicular = self.N * delta_s_perpendicular
        dr_parallel = delta_s_parallel * self.T

        D = self.create_drift_unit_vector(self.N, self.T)
        dr_drift = D * delta_s_drift

        dr = dr_parallel + dr_drift + dr_perpendicular
        dr = self.step_size * normalize(dr)
        return dr

    def _compute_update_with_fs(self, potentiostat_step_size):
        '''Uses the Frenet–Serret formulas to perform curvature based
        extrapolation to compute the update vector'''
        # this should keep the dr clear of the constraints
        # by using the actual change, not a velocity vector
        delta_r = self.r - self.rold
        delta_s = np.linalg.norm(delta_r)
        # approximation of delta_s we use this incase an adaptive step_size
        # algo get used

        delta_T = self.T - self.Told
        delta_N = self.N - self.Nold
        dTds = delta_T / delta_s
        dNds = delta_N / delta_s
        if self.use_tangent_curvature:
            curvature = np.linalg.norm(dTds)
            # on a perfect trajectory, the normal can be computed this way,
            # But the normal should always be tied to forces
            # N = dTds / curvature
        else:
            # normals are better since they are fixed to the reality of
            # forces. I see smaller forces and energy errors in bulk systems
            # using the normals for curvature
            curvature = np.linalg.norm(dNds)
        self.curvature = curvature

        if self.angle_limit is not None:
            phi = np.pi / 180 * self.angle_limit
            self.step_size = np.sqrt(2 - 2 * np.cos(phi)) / curvature
            self.step_size = min(self.step_size, self.maxstep)

        # now we can compute a safe step
        delta_s_perpendicular, delta_s_parallel, delta_s_drift = \
            self.compute_step_contributions(
                potentiostat_step_size)

        N_guess = self.N + dNds * delta_s_parallel
        T_guess = self.T + dTds * delta_s_parallel
        # the extrapolation is good at keeping N_guess and T_guess
        # orthogonal but not normalized:
        N_guess = normalize(N_guess)
        T_guess = normalize(T_guess)

        dr_perpendicular = delta_s_perpendicular * (N_guess)

        dr_parallel = delta_s_parallel * self.T * \
            (1 - (delta_s_parallel * curvature)**2 / 6.0) \
            + self.N * (curvature / 2.0) * delta_s_parallel**2

        D = self.create_drift_unit_vector(N_guess, T_guess)
        dr_drift = D * delta_s_drift

        # combine the components
        dr = dr_perpendicular + dr_parallel + dr_drift
        dr = self.step_size * normalize(dr)
        # because we guess our orthonormalization directions,
        # we renormalize to ensure a correct step size
        return dr

    def update_previous_energies(self, energy):
        '''Updates the energy history in self.previous_energies to include the
         current energy.'''
        # np.roll shifts the values to keep nice sequential ordering.
        self.previous_energies = np.roll(self.previous_energies, 1)
        self.previous_energies[0] = energy

    def compute_potentiostat_step_size(self, forces, energy):
        '''Computes the potentiostat step size by linear extrapolation of the
        potential energy using the forces. The step size can be positive or
        negative depending on whether or not the energy is too high or too low.
        '''
        if self.use_target_shift:
            target_shift = self.energy_target - np.mean(self.previous_energies)
        else:
            target_shift = 0.0

        # deltaU is the potential error that will be corrected for
        deltaU = energy - (self.energy_target + target_shift)

        f_norm = np.linalg.norm(forces)
        # can be positive or negative
        potentiostat_step_size = (deltaU / f_norm) * \
            self.potentiostat_step_scale
        return potentiostat_step_size

    def step(self, f=None):
        atoms = self._actual_atoms

        if f is None:
            f = atoms.get_forces()

        # get the velocity vector and old kinetic energy for momentum rescaling
        velocities = atoms.get_velocities()
        KEold = atoms.get_kinetic_energy()

        energy = atoms.get_potential_energy(force_consistent=True)
        self.update_previous_energies(energy)
        potentiostat_step_size = self.compute_potentiostat_step_size(f, energy)

        self.N = normalize(f)
        self.r = atoms.get_positions()
        # remove velocity  projection on forces
        v_parallel = subtract_projection(velocities, self.N)
        self.T = normalize(v_parallel)

        if self.use_frenet_serret:
            if self.Nold is not None and self.Told is not None:
                dr = self._compute_update_with_fs(potentiostat_step_size)
            else:
                # we must have the old positions and vectors for an FS step
                # if we don't, we can only do a small step
                dr = self._compute_update_without_fs(
                    potentiostat_step_size,
                    scale=self.initialization_step_scale)
        else:  # of course we can run less accuratly without FS.
            dr = self._compute_update_without_fs(potentiostat_step_size)

        # now that dr is done, we check if there is translation
        if self.remove_translation:
            net_motion = dr.sum(axis=0) / len(atoms)
            # print(net_motion)
            dr = dr - net_motion
            dr_unit = dr / np.linalg.norm(dr)
            dr = dr_unit * self.step_size

        # save old positions before update
        self.Nold = self.N
        self.rold = self.r
        self.Told = self.T

        # if we have constraints then this will do the first part of the
        # RATTLE algorithm:
        # If we can avoid using momenta, this will be simpler.
        masses = atoms.get_masses()[:, np.newaxis]
        atoms.set_positions(self.r + dr)
        new_momenta = (atoms.get_positions() - self.r) * masses  # / self.dt

        # We need to store the momenta on the atoms before calculating
        # the forces, as in a parallel Asap calculation atoms may
        # migrate during force calculations, and the momenta need to
        # migrate along with the atoms.
        atoms.set_momenta(new_momenta, apply_constraint=False)

        # Now we get the new forces!
        f = atoms.get_forces(md=True)

        # I don't really know if removing md=True from above will break
        # compatibility with RATTLE, leaving it alone for now.
        f_constrained = atoms.get_forces()
        # but this projection needs the forces to be consistent with the
        # constraints. We have to set the new velocities perpendicular so they
        # get logged properly in the trajectory files.
        vnew = subtract_projection(atoms.get_velocities(), f_constrained)
        # using the md = True forces like this:
        # vnew = subtract_projection(atoms.get_velocities(), f)
        # will not work with constraints
        atoms.set_velocities(vnew)

        # rescaling momentum to maintain constant kinetic energy.
        KEnew = atoms.get_kinetic_energy()
        Ms = np.sqrt(KEold / KEnew)  # Ms = Momentum_scale
        atoms.set_momenta(Ms * atoms.get_momenta())

        # Normally this would be the second part of RATTLE
        # will be done here like this:
        #  atoms.set_momenta(atoms.get_momenta() + 0.5 * self.dt * f)
        return f
