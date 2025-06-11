# fmt: off

from typing import IO, Any, Dict, List, Optional, Type, Union

from numpy.linalg import norm

from ase import Atoms
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.optimize.optimize import Optimizer


class BFGSClimbFixInternals(BFGS):
    """Class for transition state search and optimization

    Climbs the 1D reaction coordinate defined as constrained internal coordinate
    via the :class:`~ase.constraints.FixInternals` class while minimizing all
    remaining degrees of freedom.

    Details: Two optimizers, 'A' and 'B', are applied orthogonal to each other.
    Optimizer 'A' climbs the constrained coordinate while optimizer 'B'
    optimizes the remaining degrees of freedom after each climbing step.
    Optimizer 'A' uses the BFGS algorithm to climb along the projected force of
    the selected constraint. Optimizer 'B' can be any ASE optimizer
    (default: BFGS).

    In combination with other constraints, the order of constraints matters.
    Generally, the FixInternals constraint should come first in the list of
    constraints.
    This has been tested with the :class:`~ase.constraints.FixAtoms` constraint.

    Inspired by concepts described by P. N. Plessow. [1]_

    .. [1] Plessow, P. N., Efficient Transition State Optimization of Periodic
           Structures through Automated Relaxed Potential Energy Surface Scans.
           J. Chem. Theory Comput. 2018, 14 (2), 981–990.
           https://doi.org/10.1021/acs.jctc.7b01070.

    .. note::
       Convergence is based on 'fmax' of the total forces which is the sum of
       the projected forces and the forces of the remaining degrees of freedom.
       This value is logged in the 'logfile'. Optimizer 'B' logs 'fmax' of the
       remaining degrees of freedom without the projected forces. The projected
       forces can be inspected using the :meth:`get_projected_forces` method:

       >>> for _ in dyn.irun():
       ...     projected_forces = dyn.get_projected_forces()

    Example
    -------
    .. literalinclude:: ../../ase/test/optimize/test_climb_fix_internals.py
       :end-before: # end example for documentation
    """

    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = '-',
        trajectory: Optional[str] = None,
        maxstep: Optional[float] = None,
        alpha: Optional[float] = None,
        climb_coordinate: Optional[List[FixInternals]] = None,
        optB: Type[Optimizer] = BFGS,
        optB_kwargs: Optional[Dict[str, Any]] = None,
        optB_fmax: float = 0.05,
        optB_fmax_scaling: float = 0.0,
        **kwargs,
    ):
        """Allowed parameters are similar to the parent class
        :class:`~ase.optimize.bfgs.BFGS` with the following additions:

        Parameters
        ----------
        climb_coordinate: list
            Specifies which subconstraint of the
            :class:`~ase.constraints.FixInternals` constraint is to be climbed.
            Provide the corresponding nested list of indices
            (including coefficients in the case of Combo constraints).
            See the example above.

        optB: any ASE optimizer, optional
            Optimizer 'B' for optimization of the remaining degrees of freedom
            after each climbing step.

        optB_kwargs: dict, optional
            Specifies keyword arguments to be passed to optimizer 'B' at its
            initialization. By default, optimizer 'B' writes a logfile and
            trajectory (optB_{...}.log, optB_{...}.traj) where {...} is the
            current value of the ``climb_coordinate``. Set ``logfile`` to '-'
            for console output. Set ``trajectory`` to 'None' to suppress
            writing of the trajectory file.

        optB_fmax: float, optional
            Specifies the convergence criterion 'fmax' of optimizer 'B'.

        optB_fmax_scaling: float, optional
            Scaling factor to dynamically tighten 'fmax' of optimizer 'B' to
            the value of ``optB_fmax`` when close to convergence.
            Can speed up the climbing process. The scaling formula is

            'fmax' = ``optB_fmax`` + ``optB_fmax_scaling``
            :math:`\\cdot` norm_of_projected_forces

            The final optimization with optimizer 'B' is
            performed with ``optB_fmax`` independent of ``optB_fmax_scaling``.
        """
        self.targetvalue = None  # may be assigned during restart in self.read()
        super().__init__(atoms, restart=restart, logfile=logfile,
                         trajectory=trajectory, maxstep=maxstep,
                         alpha=alpha, **kwargs)

        self.constr2climb = get_constr2climb(
            self.optimizable.atoms, climb_coordinate)
        self.targetvalue = self.targetvalue or self.constr2climb.targetvalue

        self.optB = optB
        self.optB_kwargs = optB_kwargs or {}
        self.optB_fmax = optB_fmax
        self.scaling = optB_fmax_scaling
        # log optimizer 'B' in logfiles named after current value of constraint
        self.autolog = 'logfile' not in self.optB_kwargs
        self.autotraj = 'trajectory' not in self.optB_kwargs

    def read(self):
        (self.H, self.pos0, self.forces0, self.maxstep,
         self.targetvalue) = self.load()

    def step(self):
        self.relax_remaining_dof()  # optimization with optimizer 'B'

        pos, dpos = self.pretend2climb()  # with optimizer 'A'
        self.update_positions_and_targetvalue(pos, dpos)  # obey other constr.

        self.dump((self.H, self.pos0, self.forces0, self.maxstep,
                   self.targetvalue))

    def pretend2climb(self):
        """Get directions for climbing and climb with optimizer 'A'."""
        proj_forces = self.get_projected_forces()
        pos = self.optimizable.get_positions()
        dpos, steplengths = self.prepare_step(pos, proj_forces)
        dpos = self.determine_step(dpos, steplengths)
        return pos, dpos

    def update_positions_and_targetvalue(self, pos, dpos):
        """Adjust constrained targetvalue of constraint and update positions."""
        self.constr2climb.adjust_positions(pos, pos + dpos)  # update sigma
        self.targetvalue += self.constr2climb.sigma          # climb constraint
        self.constr2climb.targetvalue = self.targetvalue     # adjust positions
        self.optimizable.set_positions(
            self.optimizable.get_positions())   # to targetvalue

    def relax_remaining_dof(self):
        """Optimize remaining degrees of freedom with optimizer 'B'."""
        if self.autolog:
            self.optB_kwargs['logfile'] = f'optB_{self.targetvalue}.log'
        if self.autotraj:
            self.optB_kwargs['trajectory'] = f'optB_{self.targetvalue}.traj'
        fmax = self.get_scaled_fmax()
        with self.optB(self.optimizable.atoms, **self.optB_kwargs) as opt:
            opt.run(fmax)  # optimize with scaled fmax
            if self.converged() and fmax > self.optB_fmax:
                # (final) optimization with desired fmax
                opt.run(self.optB_fmax)

    def get_scaled_fmax(self):
        """Return the adaptive 'fmax' based on the estimated distance to the
        transition state."""
        return (self.optB_fmax +
                self.scaling * norm(self.constr2climb.projected_forces))

    def get_projected_forces(self):
        """Return the projected forces along the constrained coordinate in
        uphill direction (negative sign)."""
        forces = self.constr2climb.projected_forces
        forces = -forces.reshape(self.optimizable.get_positions().shape)
        return forces

    def get_total_forces(self):
        """Return forces obeying all constraints plus projected forces."""
        return self.optimizable.get_forces() + self.get_projected_forces()

    def converged(self, forces=None):
        """Did the optimization converge based on the total forces?"""
        forces = forces or self.get_total_forces()
        return super().converged(forces=forces)

    def log(self, forces=None):
        forces = forces or self.get_total_forces()
        super().log(forces=forces)


def get_fixinternals(atoms):
    """Get pointer to the FixInternals constraint on the atoms object."""
    all_constr_types = list(map(type, atoms.constraints))
    index = all_constr_types.index(FixInternals)  # locate constraint
    return atoms.constraints[index]


def get_constr2climb(atoms, climb_coordinate):
    """Get pointer to the subconstraint that is to be climbed.
    Identification by its definition via indices (and coefficients)."""
    constr = get_fixinternals(atoms)
    return constr.get_subconstraint(atoms, climb_coordinate)
