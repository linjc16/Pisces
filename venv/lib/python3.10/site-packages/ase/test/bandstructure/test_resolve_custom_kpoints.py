import numpy as np
import pytest

from ase.dft.kpoints import resolve_custom_points


@pytest.fixture()
def special_points():
    return dict(A=np.zeros(3),
                B=np.ones(3))


def test_str(special_points):
    path, dct = resolve_custom_points('AB', special_points, 0)
    assert path == 'AB'
    assert set(dct) == set('AB')


def test_recognize_points_from_coords(special_points):
    path, dct = resolve_custom_points(
        [[special_points['A'], special_points['B']]], special_points, 1e-5)
    assert path == 'AB'
    assert set(dct) == set('AB')


@pytest.mark.parametrize(
    'kptcoords',
    [
        [np.zeros(3), np.ones(3)],
        [[np.zeros(3), np.ones(3)]],
    ]
)
def test_autolabel_points_from_coords(kptcoords, special_points):
    path, dct = resolve_custom_points(kptcoords, {}, 0)
    assert path == 'Kpt0Kpt1'
    assert set(dct) == {'Kpt0', 'Kpt1'}  # automatically labelled


def test_bad_shape():
    with pytest.raises(ValueError):
        resolve_custom_points([[np.zeros(2)]], {}, 0)
