import pytest

from ase.calculators.excitation_list import polarizability
from ase.calculators.h2morse import H2Morse, H2MorseExcitedStatesCalculator


def test_shapes():
    """Test evaluation of polarizabily and resulting shapes"""
    atoms = H2Morse()
    exl = H2MorseExcitedStatesCalculator().calculate(atoms)

    alphaf = polarizability(exl, range(2))
    assert alphaf.shape == (2, )
    assert alphaf.dtype == float
    alphat = polarizability(exl, 5 + 2j, tensor=True)
    assert alphat.shape == (3, 3)
    assert alphat.dtype == complex
    alphat = polarizability(exl, range(2), tensor=True)
    assert alphat.shape == (2, 3, 3)
    assert alphat.dtype == float

    # check tensor
    for af, at in zip(alphaf, alphat):
        assert at.diagonal().sum() / 3 == pytest.approx(af, 1.e-8)
