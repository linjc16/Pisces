import numpy as np
import pytest

from ase.lattice import MCLC


@pytest.fixture()
def lat():
    return MCLC(3, 4, 5, 70)


@pytest.fixture()
def cell(lat):
    return lat.tocell()


@pytest.fixture()
def bandpath(lat):
    return lat.bandpath(npoints=0)


def test_cartesian_kpts(bandpath):
    kpts1 = bandpath.icell.cartesian_positions(bandpath.kpts)
    kpts2 = bandpath.cartesian_kpts()
    assert kpts1 == pytest.approx(kpts2)


def test_interpolate_npoints(bandpath):
    path = bandpath.interpolate(npoints=42)
    assert len(path.kpts) == 42


def test_interpolate_density(bandpath):
    path1 = bandpath.interpolate(density=10)
    path2 = bandpath.interpolate(density=20)
    assert len(path1.kpts) == len(path2.kpts) // 2


def test_zero_npoints(lat):
    path = lat.bandpath(npoints=0)
    assert path.path == lat.special_path
    assert len(path.kpts) == len(path.get_linear_kpoint_axis()[2])  # XXX ugly


@pytest.fixture()
def custom_points():
    rng = np.random.RandomState(0)
    dct = {name: rng.random(3) for name in ['K', 'K1', 'Kpoint', 'Kpoint1']}
    return dct


def test_custom_points(cell, custom_points):
    npoints = 42
    path = cell.bandpath('KK1,KpointKpoint1', special_points=custom_points,
                         npoints=npoints)

    print(path)
    assert len(path.kpts) == npoints
    assert path.kpts[0] == pytest.approx(custom_points['K'])
    assert path.kpts[-1] == pytest.approx(custom_points['Kpoint1'])


def test_autolabel_kpoints(cell):
    kpt0 = np.zeros(3)
    kpt1 = np.ones(3)
    path = cell.bandpath([[kpt0, kpt1]], npoints=17,
                         special_points={})
    assert len(path.kpts == 17)
    assert set(path.special_points) == {'Kpt0', 'Kpt1'}
    assert path.kpts[0] == pytest.approx(kpt0)
    assert path.kpts[-1] == pytest.approx(kpt1)


def test_bad_kpointlist(cell):
    with pytest.raises(ValueError):
        cell.bandpath([np.zeros(2)])
