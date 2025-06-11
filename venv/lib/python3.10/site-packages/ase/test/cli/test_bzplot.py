import pytest

from ase import Atoms
from ase.build import bulk, fcc111
from ase.io import write


@pytest.fixture(
    params=[
        bulk('Ti'),
        fcc111('Au', size=(1, 1, 1)),
        pytest.param(Atoms('H', cell=[0, 0, 1], pbc=[0, 0, 1])),
    ],
    ids=lambda atoms: f'{atoms.cell.rank}-dim',
)
def file(request, testdir):
    atoms = request.param
    file = f'atoms.{atoms.cell.rank}dim.traj'
    write(file, atoms)
    return file


def test_bzplot(cli, file, plt):
    cli.ase('reciprocal', file, 'bandpath.svg')
