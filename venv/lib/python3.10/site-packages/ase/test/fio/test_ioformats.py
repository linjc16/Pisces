import pytest

from ase.io.formats import ioformats


def test_manually():
    traj = ioformats['traj']
    print(traj)

    outcar = ioformats['vasp-out']
    print(outcar)
    assert outcar.match_name('OUTCAR')
    assert outcar.match_name('something.with.OUTCAR.stuff')


@pytest.fixture()
def excitingtools():
    """If we cannot import excitingtools we skip tests with this fixture."""
    return pytest.importorskip('excitingtools')


@pytest.mark.parametrize('name', ioformats)
def test_ioformat(name, excitingtools):
    """Test getting the full description of each ioformat."""
    ioformat = ioformats[name]
    print(name)
    print('=' * len(name))
    print(ioformat.full_description())
    print()
