import numpy as np
import pytest

from ase.geometry.bravais_type_engine import (
    generate_niggli_op_table,
    niggli_op_table,
)


def lattice_names():
    return [name for name in niggli_op_table
            if name not in {'MCL', 'MCLC', 'TRI'}]


@pytest.mark.parametrize('lattice_name', lattice_names())
def test_generate_niggli_table(lattice_name):
    length_grid = np.logspace(-1, 1, 30)
    angle_grid = np.linspace(30, 170, 50)
    table = generate_niggli_op_table(lattices=[lattice_name],
                                     angle_grid=angle_grid,
                                     length_grid=length_grid)

    thistable = table[lattice_name]
    reftable = niggli_op_table[lattice_name]

    assert set(thistable) == set(reftable)
