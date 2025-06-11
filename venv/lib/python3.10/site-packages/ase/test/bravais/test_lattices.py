import numpy as np
import pytest

from ase.lattice import (
    all_variants,
    bravais_lattices,
    get_lattice_from_canonical_cell,
)


@pytest.mark.parametrize('name', list(bravais_lattices))
def test_bravais_lattices(name):
    latcls = bravais_lattices[name]
    assert latcls.name == name
    assert latcls.longname is not None
    for par in latcls.parameters:
        assert par in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']


@pytest.mark.parametrize('lat', list(all_variants()),
                         ids=lambda lat: lat.variant)
def test_variants(lat):
    print(lat.variant)
    for par in lat.parameters:
        print(par, getattr(lat, par))

    print('cell', lat.tocell())
    cell = lat.tocell()
    if lat.name in ['TRI']:
        # Automatic check not implemented for these cell types, but we
        # can still recognize the canonical form:
        lat1 = get_lattice_from_canonical_cell(cell)
    else:
        lat1 = cell.get_bravais_lattice()
    assert lat1.name == lat.name, (lat1.name, lat.name)
    assert lat1.variant == lat.variant
    assert np.abs(cell - lat1.tocell()).max() < 1e-13
    print('cellpar', lat.cellpar())
    print('special path', lat.special_path)
    arr = lat.get_special_points_array()
    assert arr.shape == (len(lat.special_point_names), 3)

    dct = lat.get_special_points()
    assert len(dct) == len(lat.special_point_names)
    print(lat)
