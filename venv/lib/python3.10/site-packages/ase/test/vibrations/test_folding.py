import pytest

from ase.build import molecule
from ase.test.utils import RandomCalculator
from ase.vibrations import Infrared


def test_folding(testdir):
    """Test that folding is consitent with intensities"""

    atoms = molecule('C2H6')
    atoms.calc = RandomCalculator()
    ir = Infrared(atoms)
    ir.run()
    freqs = ir.get_frequencies().real

    for folding in ['Gaussian', 'Lorentzian']:
        x, y = ir.get_spectrum(start=freqs.min() - 100,
                               end=freqs.max() + 100,
                               type=folding,
                               normalize=True)
        assert ir.intensities.sum() == pytest.approx(
            y.sum() * (x[1] - x[0]), 1e-2)
