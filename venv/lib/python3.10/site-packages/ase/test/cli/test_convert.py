from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.io import read, write


def test_convert(tmp_path, cli):
    infile = tmp_path / 'images.json'
    images = [bulk('Si'), bulk('Au')]
    write(infile, images, format='json')

    outfile = tmp_path / 'images.traj'
    cli.ase('convert', str(infile), str(outfile))
    images2 = read(outfile, ':')

    assert len(images2) == 2
    for a1, a2 in zip(images, images2):
        assert not compare_atoms(a1, a2)
