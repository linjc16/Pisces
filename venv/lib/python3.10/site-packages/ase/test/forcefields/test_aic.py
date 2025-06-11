import numpy as np

from ase import Atoms
from ase.calculators.counterions import AtomicCounterIon as ACI
from ase.calculators.fd import calculate_numerical_forces


def test_aic():
    """Test Atomic Counter Ion calc forces."""

    atoms = Atoms('2Na', positions=np.array([[0, 0, 0], [0, 0, 4]]))

    atoms.calc = ACI(1, 1.6642, 0.0001201186, rc=4.5)
    f = atoms.get_forces()
    df = calculate_numerical_forces(atoms, 1e-6) - f
    print(df)
    assert abs(df).max() < 2e-6
