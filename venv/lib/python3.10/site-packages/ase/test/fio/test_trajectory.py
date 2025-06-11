import pytest

from ase import Atom, Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.constraints import FixBondLength
from ase.io import Trajectory, read


@pytest.fixture()
def co():
    return Atoms([Atom('C', (0, 0, 0)),
                  Atom('O', (0, 0, 1.2))])


@pytest.fixture()
def trajfile_and_images(co):
    fname = '1.traj'
    co = co.copy()

    with Trajectory(fname, 'w', co) as traj:
        written = []

        for _ in range(5):
            co.positions[:, 2] += 0.1
            traj.write()
            written.append(co.copy())

    with Trajectory('1.traj', 'a') as traj:
        co = read('1.traj')
        print(co.positions)
        co.positions[:] += 1
        traj.write(co)
        written.append(co.copy())

    with Trajectory('1.traj') as traj:
        for a in traj:
            print(1, a.positions[-1, 2])

    co.positions[:] += 1

    with Trajectory('1.traj', 'a') as traj:
        traj.write(co)
        written.append(co.copy())
        assert len(traj) == 7

        co[0].number = 1
        traj.write(co)
        written.append(co.copy())

        co[0].number = 6
        co.pbc = True
        traj.write(co)
        written.append(co.copy())

        co.pbc = False
        o = co.pop(1)
        traj.write(co)
        written.append(co.copy())

        co.append(o)
        traj.write(co)
        written.append(co.copy())

    return fname, written


@pytest.fixture()
def trajfile(trajfile_and_images):
    return trajfile_and_images[0]


@pytest.fixture()
def images(trajfile_and_images):
    return trajfile_and_images[1]


def test_trajectory(trajfile, images):
    imgs = read(trajfile, index=':')
    assert len(imgs) == len(images)
    for img1, img2 in zip(imgs, images):
        assert img1 == img2

    # Verify slicing works.
    with Trajectory(trajfile, 'r') as read_traj:
        sliced_traj = read_traj[3:8]
        assert len(sliced_traj) == 5
        sliced_again = sliced_traj[1:-1]
        assert len(sliced_again) == 3
        assert sliced_traj[1] == sliced_again[0]


def test_append_nonexistent_file(co):
    fname = '2.traj'
    with Trajectory(fname, 'a', co) as t:
        pass

    with Trajectory('empty.traj', 'w') as t:
        pass

    with Trajectory('empty.traj', 'r') as t:
        assert len(t) == 0


def test_only_energy():
    with Trajectory('fake.traj', 'w') as t:
        t.write(Atoms('H'), energy=-42.0, forces=[[1, 2, 3]])

    a = read('fake.traj')
    with Trajectory('only-energy.traj', 'w', properties=['energy']) as t:
        t.write(a)

    b = read('only-energy.traj')
    e = b.get_potential_energy()
    assert e + 42 == 0
    with pytest.raises(PropertyNotImplementedError):
        b.get_forces()


def test_constraint_and_momenta():
    a = Atoms('H2',
              positions=[(0, 0, 0), (0, 0, 1)],
              momenta=[(1, 0, 0), (0, 0, 0)])
    a.constraints = [FixBondLength(0, 1)]
    with Trajectory('constraint.traj', 'w', a) as t:
        t.write()
    b = read('constraint.traj')
    assert not (b.get_momenta() - a.get_momenta()).any()
