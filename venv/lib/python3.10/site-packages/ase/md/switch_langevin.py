# fmt: off

from typing import Any, List, Optional

import numpy as np
from scipy.integrate import trapezoid

from ase import Atoms
from ase.calculators.mixing import MixedCalculator
from ase.md.langevin import Langevin


class SwitchLangevin(Langevin):
    """
    MD class for carrying out thermodynamic integration between two
    hamiltonians

    Read more at https://en.wikipedia.org/wiki/Thermodynamic_integration

    The integration path is lambda 0 ---> 1 , with the switching hamiltonian
    H(lambda) = (1-lambda) * H1 + lambda * H2

    Attributes
    ----------
    path_data : numpy array
        col 1 (md-step), col 2 (lambda), col 3 (energy H1), col 4 (energy H2)

    Parameters
    ----------
    atoms : ASE Atoms object
        Atoms object for which MD will be run
    calc1 : ASE calculator object
        Calculator corresponding to first Hamiltonian
    calc2 : ASE calculator object
        Calculator corresponding to second Hamiltonian
    dt : float
        Timestep for MD simulation
    T : float
        Temperature in eV (deprecated)
    friction : float
        Friction for langevin dynamics
    n_eq : int
        Number of equilibration steps
    n_switch : int
        Number of switching steps
    """

    def __init__(
        self,
        atoms: Atoms,
        calc1,
        calc2,
        dt: float,
        T: Optional[float] = None,
        friction: Optional[float] = None,
        n_eq: Optional[int] = None,
        n_switch: Optional[int] = None,
        temperature_K: Optional[float] = None,
        **langevin_kwargs,
    ):
        super().__init__(atoms, dt, temperature=T, temperature_K=temperature_K,
                         friction=friction, **langevin_kwargs)
        if friction is None:
            raise TypeError("Missing 'friction' argument.")
        if n_eq is None:
            raise TypeError("Missing 'n_eq' argument.")
        if n_switch is None:
            raise TypeError("Missing 'n_switch' argument.")
        self.n_eq = n_eq
        self.n_switch = n_switch
        self.lam = 0.0
        calc = MixedCalculator(calc1, calc2, weight1=1.0, weight2=0.0)
        self.atoms.calc = calc

        self.path_data: List[Any] = []

    def run(self):
        """ Run the MD switching simulation """
        forces = self.atoms.get_forces(md=True)

        # run equilibration with calc1
        for _ in range(self.n_eq):
            forces = self.step(forces)
            self.nsteps += 1
            self.call_observers()

        # run switch from calc1 to calc2
        self.path_data.append(
            [0, self.lam,
             *self.atoms.calc.get_energy_contributions(self.atoms)])
        for step in range(1, self.n_switch):
            # update calculator
            self.lam = get_lambda(step, self.n_switch)
            self.atoms.calc.set_weights(1 - self.lam, self.lam)

            # carry out md step
            forces = self.step(forces)
            self.nsteps += 1

            # collect data
            self.call_observers()
            self.path_data.append(
                [step, self.lam,
                 *self.atoms.calc.get_energy_contributions(self.atoms)])

        self.path_data = np.array(self.path_data)

    def get_free_energy_difference(self):
        """ Return the free energy difference between calc2 and calc1, by
        integrating dH/dlam along the switching path

        Returns
        -------
        float
            Free energy difference, F2 - F1
        """
        if len(self.path_data) == 0:
            raise ValueError('No free energy data found.')

        lambdas = self.path_data[:, 1]
        U1 = self.path_data[:, 2]
        U2 = self.path_data[:, 3]
        delta_F = trapezoid(U2 - U1, lambdas)
        return delta_F


def get_lambda(step, n_switch):
    """ Return lambda value along the switching path """
    assert step >= 0 and step <= n_switch
    t = step / (n_switch - 1)
    return t**5 * (70 * t**4 - 315 * t**3 + 540 * t**2 - 420 * t + 126)
