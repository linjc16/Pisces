import pytest

from ase.build import bulk
from ase.io import write


def test_info(cli):
    assert 'numpy' in cli.ase('info')


def test_info_formats(cli):
    assert 'traj' in cli.ase('info', '--formats')


def test_info_calculators(cli):
    # The configuration listing will contain all configurable calculators
    # whether they are configured or not.
    assert 'nwchem' in cli.ase('info', '--calculators')


@pytest.fixture()
def fname(testdir):
    atoms = bulk('Au')
    filename = 'file.traj'
    write(filename, atoms)
    return filename


def test_info_file_ok(cli, fname):
    assert 'trajectory' in cli.ase('info', '--files', fname)


def test_info_file_fail(cli):
    cli.ase('info', '--files', 'nonexistent_file.traj', expect_fail=True)
