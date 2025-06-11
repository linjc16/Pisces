import numpy as np
import pytest

from ase.dft.bee import BEEFEnsemble, ensemble, readbee


class BEECalculator:
    """Fake DFT calculator."""
    atoms = None

    def __init__(self, name):
        self.name = name

    def get_xc_functional(self):
        return self.name

    def get_nonselfconsistent_energies(self, beef_type: str) -> np.ndarray:
        n = {'mbeef': 64,
             'beefvdw': 32,
             'mbeefvdw': 28}[beef_type]
        return np.linspace(-1, 1, n)

    def get_potential_energy(self, atoms):
        return 0.0


@pytest.mark.parametrize('xc', ['mBEEF', 'BEEF-vdW', 'mBEEF-vdW'])
def test_bee(xc, testdir):
    """Check BEEF ensemble code."""
    size = 7  # size of ensemble

    # From a calculator:
    calc = BEECalculator(xc)
    ens = BEEFEnsemble(calc)
    energies = ens.get_ensemble_energies(size)
    assert energies.shape == (size,)

    # From a file:
    ens.write(f'{xc}.bee')
    e, de, contribs, _seed, xc = readbee(f'{xc}.bee', all=True)
    assert e + de == pytest.approx(energies, abs=1e-12)
    e2000 = ensemble(e, contribs, xc)
    assert e2000.shape == (2000,)

    # From data:
    ens = BEEFEnsemble(e=e, contribs=contribs, xc=xc, verbose=False)
    energies2 = ens.get_ensemble_energies(size)
    assert energies2 == pytest.approx(energies, abs=1e-12)
