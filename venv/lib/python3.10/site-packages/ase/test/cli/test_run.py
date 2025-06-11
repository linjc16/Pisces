import pytest

from ase.build import bulk
from ase.io import write


@pytest.fixture()
def fname(testdir):
    atoms = bulk('Au')
    filename = 'file.traj'
    write(filename, atoms)
    return filename


def test_run_eos(cli, fname):
    output = cli.ase('run', 'emt', fname, '--equation-of-state=4,2.0')
    print(output)
    for line in output.splitlines():
        if line.startswith('fitted volume'):
            vol = float(line.split()[-1])
            assert vol == pytest.approx(16.7, abs=0.05)
            return
    raise ValueError('Volume not found')
