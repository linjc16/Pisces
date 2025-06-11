import pytest

from ase.build import bulk, molecule
from ase.io import write


@pytest.fixture()
def atoms():
    return bulk('Au')


@pytest.fixture()
def fname(atoms, testdir):
    filename = 'file.traj'
    write(filename, atoms)
    return filename


def test_exec_fail_withoutcode(cli, fname):
    cli.ase('exec', fname, expect_fail=True)


def test_exec_atoms(cli, fname, atoms):
    out = cli.ase('exec', fname, '-e', 'print(atoms.symbols)')
    assert out.strip() == str(atoms.symbols)


def test_exec_index(cli, fname):
    out = cli.ase('exec', fname, '-e', 'print(index)')
    assert out.strip() == str(0)


def test_exec_images(cli, fname, atoms):
    out = cli.ase('exec', fname, '-e', 'print(len(images[0]))')
    assert out.strip() == str(len(atoms))


@pytest.fixture()
def images():
    images = []
    for name in ['C6H6', 'H2O', 'CO']:
        images.append(molecule(name))
    return images


@pytest.fixture()
def fnameimages(images, testdir):
    filename = 'fileimgs.xyz'
    write(filename, images)
    return filename


@pytest.fixture()
def execfilename(testdir):
    filename = 'execcode.py'
    with open(filename, 'w') as fd:
        fd.write('print(len(atoms))')
    return filename


def test_exec_file(cli, images, fnameimages, execfilename):
    out = cli.ase('exec', fnameimages, '-E', execfilename)
    out_expected = [str(len(atoms)) for atoms in images]
    assert out.split() == out_expected
