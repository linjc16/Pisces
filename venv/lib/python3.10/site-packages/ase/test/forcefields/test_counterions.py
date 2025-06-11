import numpy as np

from ase import Atoms, units
from ase.calculators.counterions import AtomicCounterIon as ACI
from ase.calculators.fd import calculate_numerical_forces


def test_counterions():
    """ Test AtomicCounterIon is force/energy consistent over
        PBCs and with cutoff """

    sigma = 1.868 * (1.0 / 2.0)**(1.0 / 6.0)
    epsilon = 0.00277 * units.kcal / units.mol

    atoms = Atoms('3Na', positions=np.array([[0, 0, -2], [0, 0, 0], [0, 0, 2]]))
    atoms.cell = [10, 10, 10]
    atoms.pbc = True

    atoms.calc = ACI(1, epsilon, sigma, rc=4.5)
    points = np.arange(-15., 15., 0.2)

    for _ in points:
        f = atoms.get_forces()
        fn = calculate_numerical_forces(atoms, 1e-5)
        df = (f - fn)
        assert abs(df).max() < 1e-8
