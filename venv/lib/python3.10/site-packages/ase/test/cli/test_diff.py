import re

import pytest

from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.cli.template import (
    MapFormatter,
    Table,
    TableFormat,
    prec_round,
    slice_split,
    sym2num,
)
from ase.constraints import FixAtoms, FixedPlane
from ase.io import read
from ase.optimize import QuasiNewton


@pytest.fixture(scope="module")
def traj(tmp_path_factory):
    slab = fcc100('Al', size=(2, 2, 3))
    add_adsorbate(slab, 'Au', 1.7, 'hollow')
    slab.center(axis=2, vacuum=4.0)
    mask = [atom.tag > 1 for atom in slab]
    fixlayers = FixAtoms(mask=mask)
    plane = FixedPlane(-1, (1, 0, 0))
    slab.set_constraint([fixlayers, plane])
    slab.calc = EMT()

    temp_path = tmp_path_factory.mktemp("data")
    trajectory = temp_path / 'AlAu.traj'
    with QuasiNewton(slab, trajectory=str(trajectory)) as qn:
        qn.run(fmax=0.02)
    return str(trajectory)


def test_singleFile_falseCalc_multipleImages(cli, traj):
    stdout = cli.ase('diff', '--as-csv', traj)

    r = c = -1
    for rowcount, row in enumerate(stdout.split('\n')):
        for colcount, col in enumerate(row.split(',')):
            if col == 'Δx':
                r = rowcount + 2
                c = colcount
            if (rowcount == r) & (colcount == c):
                val = col
                break
    assert float(val) == 0.


def test_singleFile_trueCalc_multipleImages(cli, traj):
    cli.ase('diff', traj, '-c')


def test_twoFiles_falseCalc_singleImage(cli, traj):
    cli.ase('diff', f'{traj}@:1', f'{traj}@1:2')


def test_twoFiles_trueCalc_singleImage(cli, traj):
    cli.ase('diff', f'{traj}@:1', f'{traj}@1:2', '-c')


def test_twoFiles_falseCalc_multipleImages(cli, traj):
    cli.ase('diff', f'{traj}@:2', f'{traj}@2:4')


def test_twoFiles_trueCalc_multipleImages(cli, traj):
    stdout = cli.ase('diff', f'{traj}@:2', f'{traj}@2:4', '-c',
                     '--rank-order', 'dfx', '--as-csv')
    stdout = [row.split(',') for row in stdout.split('\n')]
    stdout = [row for row in stdout if len(row) > 4]

    header = stdout[0]
    body = stdout[1:len(stdout) // 2 - 1]  # note tables are appended in stdout
    for c in range(len(header)):
        if header[c] == 'Δfx':
            break
    dfx_ordered = [float(row[c]) for row in body]
    for i in range(len(dfx_ordered) - 2):
        assert dfx_ordered[i] <= dfx_ordered[i + 1]


def test_cli_opt(cli, traj):
    # template command line options
    stdout = cli.ase('diff', f'{traj}@:1', f'{traj}@:2', '-c',
                     '--template', 'p1x,p2x,dx,f1x,f2x,dfx')
    stdout = stdout.split('\n')

    for counter, row in enumerate(stdout):
        if '=' in row:  # default toprule
            header = stdout[counter + 1]
            break
    header = re.sub(r'\s+', ',', header).split(',')[1:-1]
    assert header == ['p1x', 'p2x', 'Δx', 'f1x', 'f2x', 'Δfx']

    cli.ase('diff', traj, '-c', '--template',
            'p1x,f1x,p1y,f1y:0:-1,p1z,f1z,p1,f1',
            '--max-lines', '6', '--summary-functions', 'rmsd')


def test_template_functions():
    """Test functions used in the template module."""
    num = 1.55749
    rnum = [prec_round(num, i) for i in range(1, 6)]
    assert rnum == [1.6, 1.56, 1.557, 1.5575, 1.55749]
    assert slice_split('a@1:3:1') == ('a', slice(1, 3, 1))

    sym = 'H'
    num = sym2num[sym]
    mf = MapFormatter().format
    sym2 = mf('{:h}', num)
    assert sym == sym2


def test_template_classes(traj):
    prec = 4
    tableformat = TableFormat(precision=prec, representation='f', midrule='|')
    table = Table(field_specs=('dx', 'dy', 'dz'), tableformat=tableformat)
    images = read(traj, ':')
    table_out = table.make(images[0], images[1]).split('\n')
    for counter, row in enumerate(table_out):
        if '|' in row:
            break

    row = table_out[counter + 2]

    assert 'E' not in table_out[counter + 2]

    row = re.sub(r'\s+', ',', table_out[counter + 2]).split(',')[1:-1]
    assert len(row[0]) >= prec
