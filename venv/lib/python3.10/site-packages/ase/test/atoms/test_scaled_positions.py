from ase import Atoms


def test_scaled_positions():
    assert Atoms('X', [(-1e-35, 0, 0)],
                 pbc=True).get_scaled_positions()[0, 0] < 1
