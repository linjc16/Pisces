# fmt: off

"""
This module contains tools for preconditioned geometry optimisation.

Code maintained by James Kermode <james.kermode@gmail.com>
Parts written by John Woolley, Letif Mones and Christoph Ortner.

The preconditioned LBFGS optimizer implemented here is described in
the following publication:

    D. Packwood, J. R. Kermode, L. Mones, N. Bernstein, J. Woolley,
    N. Gould, C. Ortner, and G. Csanyi, A universal preconditioner for
    simulating condensed phase materials, J. Chem. Phys. 144, 164109 (2016).
    DOI: https://doi.org/10.1063/1.4947024

A preconditioned version of FIRE is also included, this is less well tested.

Optional dependencies
---------------------

    - scipy, `pip install scipy` for efficient sparse linear algebra,
      important for large systems (>1000 atoms).
    - PyAMG, `pip install pyamg`, for iterative adaptive multi grid
      inversion of the preconditioner, again important for large systems.
"""

from ase.optimize.ode import ODE12r
from ase.optimize.precon.fire import PreconFIRE
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.optimize.precon.precon import (
    C1,
    FF,
    Exp,
    Exp_FF,
    Pfrommer,
    Precon,
    PreconImages,
    SplineFit,
    make_precon,
)


class PreconODE12r(ODE12r):
    """
    Subclass of ase.optimize.ode.ODE12r with 'Exp' preconditioning on by default
    """

    def __init__(self, *args, **kwargs):
        if 'precon' not in kwargs:
            kwargs['precon'] = 'Exp'
        ODE12r.__init__(self, *args, **kwargs)


__all__ = ['make_precon', 'PreconImages', 'SplineFit',
           'Precon', 'Exp', 'C1', 'Pfrommer', 'FF', 'Exp_FF',
           'PreconLBFGS', 'PreconFIRE', 'PreconODE12r']
