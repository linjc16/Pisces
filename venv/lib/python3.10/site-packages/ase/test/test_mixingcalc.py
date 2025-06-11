import numpy as np
import pytest

from ase.build import fcc111
from ase.calculators.calculator import CalculatorSetupError
from ase.calculators.emt import EMT
from ase.calculators.mixing import (
    AverageCalculator,
    LinearCombinationCalculator,
    MixedCalculator,
    Mixer,
    SumCalculator,
)
from ase.constraints import FixAtoms


def test_mixingcalc():
    """This test checks the basic functionality of the MixingCalculators.
    The example system is based on the SinglePointCalculator test case.
    """

    # Calculate reference values:
    atoms = fcc111("Cu", (2, 2, 1), vacuum=10.0)
    atoms[0].x += 0.2

    # First run the test with EMT similarly to the test of the single point
    # calculator.
    calc = EMT()
    atoms.calc = calc
    forces = atoms.get_forces()
    voigt_stress = atoms.get_stress()
    stress_tensor = atoms.get_stress(voigt=False)

    # Mixer: check that stresses of different shapes are handled correctly
    stresses = [voigt_stress, stress_tensor]
    reshape_stress = Mixer.make_stress_voigt(stresses)
    assert np.isclose(reshape_stress[0], reshape_stress[1]).all()

    # SumCalculator: Only one way to associate a calculator with an atoms
    # object.
    atoms1 = atoms.copy()
    calc1 = SumCalculator([EMT(), EMT()])
    atoms1.calc = calc1

    # Check the results.
    assert np.isclose(2 * forces, atoms1.get_forces()).all()
    assert np.isclose(2 * voigt_stress, atoms1.get_stress()).all()
    assert atoms1.get_stress().shape == (6,)

    # testing  step
    atoms1[0].x += 0.2
    assert not np.isclose(2 * forces, atoms1.get_forces()).all()

    # Check constraints
    atoms1.set_constraint(FixAtoms(indices=[atom.index for atom in atoms]))
    assert np.isclose(0, atoms1.get_forces()).all()

    # AverageCalculator:

    atoms1 = atoms.copy()
    calc1 = AverageCalculator([EMT(), EMT()])
    atoms1.calc = calc1

    # LinearCombinationCalculator:

    atoms2 = atoms.copy()
    calc2 = LinearCombinationCalculator([EMT(), EMT()], weights=[0.5, 0.5])
    atoms2.calc = calc2

    # Check the results (it should be the same because it is tha average of
    # the same values).
    assert np.isclose(forces, atoms1.get_forces()).all()
    assert np.isclose(forces, atoms2.get_forces()).all()

    # testing  step
    atoms1[0].x += 0.2
    assert not np.isclose(2 * forces, atoms1.get_forces()).all()

    with pytest.raises(CalculatorSetupError):
        calc1 = LinearCombinationCalculator([], [])

    with pytest.raises(CalculatorSetupError):
        calc1 = AverageCalculator([])

    # test  MixedCalculator and energy contributions
    w1, w2 = 0.78, 0.22
    atoms1 = atoms.copy()
    atoms1.calc = EMT()
    E_tot = atoms1.get_potential_energy()

    calc1 = MixedCalculator(EMT(), EMT(), w1, w2)
    E1, E2 = calc1.get_energy_contributions(atoms1)
    assert np.isclose(E1, E_tot)
    assert np.isclose(E2, E_tot)
