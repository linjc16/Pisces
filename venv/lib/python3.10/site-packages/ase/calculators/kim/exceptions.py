# fmt: off

"""
Exceptions for the general error types that can occur either while
setting up the calculator, which requires constructing KIM API C++
objects, or while running a simulation
"""
from ase.calculators.calculator import CalculatorError


class KIMCalculatorError(CalculatorError):
    """
    Indicates an error occurred in initializing an applicable
    calculator.  This either results from incompatible combinations of
    argument values passed to kim.KIM(), or from models that are
    incompatible in some way with this calculator
    """


class KIMModelNotFound(CalculatorError):
    """
    Requested model cannot be found in any of the KIM API model
    collections on the system
    """


class KIMModelInitializationError(CalculatorError):
    """
    KIM API Model object or ComputeArguments object could not be
    successfully created
    """


class KimpyError(CalculatorError):
    """
    A call to a kimpy function resulted in a RuntimeError being raised
    """


class KIMModelParameterError(CalculatorError):
    """
    An attempt to either access or mutate a KIM portable model
    parameter was unsuccessful
    """
