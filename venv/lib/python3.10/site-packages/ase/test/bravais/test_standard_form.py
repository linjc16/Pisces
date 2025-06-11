"""Bravais lattice type check.

1) For each Bravais variant, check that we recognize the
standard cell correctly.

2) For those Bravais lattices that we can recognize in non-standard form,
   Niggli-reduce them and recognize them as well."""

import numpy as np
import pytest

from ase.lattice import (
    all_variants,
    get_lattice_from_canonical_cell,
    identify_lattice,
)

variants = [lat for lat in all_variants() if lat.ndim == 3]


@pytest.mark.parametrize('lat', variants)
def test_lattice(lat):
    cell = lat.tocell()

    def check(lat1):
        print('check', repr(lat), '-->', repr(lat1))
        err = np.abs(cell.cellpar() - lat1.cellpar()).max()
        assert err < 1e-5, err

    check(get_lattice_from_canonical_cell(cell))

    if lat.name == 'TRI':
        # The TRI lattices generally permute (the ones we produce as
        # all_variants() are reduced to a form with smaller
        # orthogonality defect) which might be desirable but would
        # trigger an error in this test.
        return

    stdcell, _op = identify_lattice(cell, 1e-4)
    check(stdcell)
    rcell, _op = cell.niggli_reduce()
    stdcell, _op = identify_lattice(rcell, 1e-4)
    check(stdcell)
