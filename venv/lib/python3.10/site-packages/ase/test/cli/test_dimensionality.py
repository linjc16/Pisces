import pytest

import ase.build
from ase.io import write


def build_layer():
    atoms = ase.build.mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19)
    atoms.cell[2, 2] = 7
    atoms.set_pbc((1, 1, 1))
    return atoms


@pytest.fixture(
    params=[
        (build_layer(), 'layer'),
        (ase.build.bulk('Ti'), 'bulk'),
    ],
    ids=['layer', 'bulk'],
)
def file(request, testdir):
    atoms, dimtype = request.param
    file = f'atoms.{dimtype}.cfg'
    write(file, atoms)
    return file


@pytest.mark.parametrize("display_all", [False, True])
def test_single(cli, file, display_all):
    if display_all:
        output = cli.ase('dimensionality', '--display-all', file)
    else:
        output = cli.ase('dimensionality', file)

    rows = output.split('\n')
    rows = [line for line in rows if len(line) > 1]
    assert len(rows) >= 3
    rows = rows[2:]

    row = rows[0].split()
    if 'layer' in file:
        assert row[1] == '2D'
    elif 'bulk' in file:
        assert row[1] == '3D'

    if display_all:
        assert len(rows) > 1
