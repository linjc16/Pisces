# fmt: off

from ase.calculators.calculator import (
    BaseCalculator,
    CalculatorSetupError,
    all_changes,
)
from ase.stress import full_3x3_to_voigt_6_stress


class Mixer:
    def __init__(self, calcs, weights):
        self.check_input(calcs, weights)
        common_properties = set.intersection(
            *(set(calc.implemented_properties) for calc in calcs)
        )
        self.implemented_properties = list(common_properties)
        self.calcs = calcs
        self.weights = weights

    @staticmethod
    def check_input(calcs, weights):
        if len(calcs) == 0:
            raise CalculatorSetupError("Please provide a list of Calculators")
        if len(weights) != len(calcs):
            raise ValueError(
                "The length of the weights must be the same as"
                " the number of Calculators!"
            )

    def get_properties(self, properties, atoms):
        results = {}

        def get_property(prop):
            contribs = [calc.get_property(prop, atoms) for calc in self.calcs]
            # ensure that the contribution shapes are the same for stress prop
            if prop == "stress":
                shapes = [contrib.shape for contrib in contribs]
                if not all(shape == shapes[0] for shape in shapes):
                    if prop == "stress":
                        contribs = self.make_stress_voigt(contribs)
                    else:
                        raise ValueError(
                            f"The shapes of the property {prop}"
                            " are not the same from all"
                            " calculators"
                        )
            results[f"{prop}_contributions"] = contribs
            results[prop] = sum(
                weight * value for weight, value in zip(self.weights, contribs)
            )

        for prop in properties:  # get requested properties
            get_property(prop)
        for prop in self.implemented_properties:  # cache all available props
            if all(prop in calc.results for calc in self.calcs):
                get_property(prop)
        return results

    @staticmethod
    def make_stress_voigt(stresses):
        new_contribs = []
        for contrib in stresses:
            if contrib.shape == (6,):
                new_contribs.append(contrib)
            elif contrib.shape == (3, 3):
                new_cont = full_3x3_to_voigt_6_stress(contrib)
                new_contribs.append(new_cont)
            else:
                raise ValueError(
                    "The shapes of the stress"
                    " property are not the same"
                    " from all calculators"
                )
        return new_contribs


class LinearCombinationCalculator(BaseCalculator):
    """Weighted summation of multiple calculators."""

    def __init__(self, calcs, weights):
        """Implementation of sum of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        weights: list of float
            Weights for each calculator in the list.
        """
        super().__init__()
        self.mixer = Mixer(calcs, weights)
        self.implemented_properties = self.mixer.implemented_properties

    def calculate(self, atoms, properties, system_changes):
        """Calculates all the specific property for each calculator and
        returns with the summed value.

        """
        self.atoms = atoms.copy()  # for caching of results
        self.results = self.mixer.get_properties(properties, atoms)

    def __str__(self):
        calculators = ", ".join(
            calc.__class__.__name__ for calc in self.mixer.calcs
        )
        return f"{self.__class__.__name__}({calculators})"


class MixedCalculator(LinearCombinationCalculator):
    """
    Mixing of two calculators with different weights

    H = weight1 * H1 + weight2 * H2

    Has functionality to get the energy contributions from each calculator

    Parameters
    ----------
    calc1 : ASE-calculator
    calc2 : ASE-calculator
    weight1 : float
        weight for calculator 1
    weight2 : float
        weight for calculator 2
    """

    def __init__(self, calc1, calc2, weight1, weight2):
        super().__init__([calc1, calc2], [weight1, weight2])

    def set_weights(self, w1, w2):
        self.mixer.weights[0] = w1
        self.mixer.weights[1] = w2

    def get_energy_contributions(self, atoms=None):
        """Return the potential energy from calc1 and calc2 respectively"""
        self.calculate(
            properties=["energy"],
            atoms=atoms,
            system_changes=all_changes
        )
        return self.results["energy_contributions"]


class SumCalculator(LinearCombinationCalculator):
    """SumCalculator for combining multiple calculators.

    This calculator can be used when there are different calculators
    for the different chemical environment or for example during delta
    leaning. It works with a list of arbitrary calculators and
    evaluates them in sequence when it is required.  The supported
    properties are the intersection of the implemented properties in
    each calculator.

    """

    def __init__(self, calcs):
        """Implementation of sum of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        """

        weights = [1.0] * len(calcs)
        super().__init__(calcs, weights)


class AverageCalculator(LinearCombinationCalculator):
    """AverageCalculator for equal summation of multiple calculators (for
    thermodynamic purposes)."""

    def __init__(self, calcs):
        """Implementation of average of calculators.

        calcs: list
            List of an arbitrary number of :mod:`ase.calculators` objects.
        """
        n = len(calcs)

        if n == 0:
            raise CalculatorSetupError(
                "The value of the calcs must be a list of Calculators"
            )

        weights = [1 / n] * n
        super().__init__(calcs, weights)
