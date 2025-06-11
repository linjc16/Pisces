from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.fd import calculate_numerical_forces


def test_h2():

    h2 = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1.1)],
               calculator=EMT())
    f1 = calculate_numerical_forces(h2, 0.0001)
    f2 = h2.get_forces()
    assert abs(f1 - f2).max() < 1e-6
