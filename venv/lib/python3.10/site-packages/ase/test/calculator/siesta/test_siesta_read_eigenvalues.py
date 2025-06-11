import pytest

from ase.build import bulk
from ase.io.siesta_output import OutputReader


def test_siesta_read_eigenvalues_soc(datadir, config_file, tmp_path):
    """ In this test, we read a stored siesta.EIG file."""
    reader = OutputReader(prefix='siesta', directory=tmp_path)
    assert reader.read_eigenvalues() == {}

    reader = OutputReader(prefix='siesta', directory=datadir / 'siesta')
    dct = reader.read_eigenvalues()
    assert dct['eigenvalues'].shape == (1, 1, 30)


@pytest.mark.calculator('siesta')
def test_siesta_read_eigenvalues(factory):
    # Test real calculation which produces a gapped .EIG file
    atoms = bulk('Si')
    calc = factory.calc(kpts=[2, 1, 1])
    atoms.calc = calc
    atoms.get_potential_energy()

    assert calc.results['eigenvalues'].shape[:2] == (1, 2)  # spins x kpts
    assert calc.get_k_point_weights().shape == (2,)
    assert calc.get_ibz_k_points().shape == (2, 3)
    assert calc.get_number_of_spins() == 1
