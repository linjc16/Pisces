from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.io.bytes import parse_atoms, parse_images, to_bytes

atoms = bulk('Ti')
images = [bulk('Au'), bulk('Ti'), bulk('NaCl', 'rocksalt', 17)]


def test_format():
    fmt = 'traj'  # TODO: parametrize?
    buf = to_bytes(atoms, format=fmt)
    atoms1 = parse_atoms(buf)

    err = compare_atoms(atoms, atoms1)
    assert not err, err  # Should be empty list

    buf = to_bytes(images, format=fmt)
    images1 = parse_images(buf)
    assert len(images) == len(images1)
    for img, img1 in zip(images, images1):
        err = compare_atoms(img, img1)
        assert not err, err
