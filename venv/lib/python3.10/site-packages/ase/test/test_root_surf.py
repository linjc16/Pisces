import pytest

from ase.build import (
    bcc111,
    bcc111_root,
    fcc111,
    fcc111_root,
    hcp0001,
    hcp0001_root,
    root_surface,
    root_surface_analysis,
)


def test_root_surf():

    # Manually checked set of roots for FCC111
    fcc111_21_set = {1, 3, 4, 7, 9, 12, 13, 16, 19, 21}

    # Keep pairs for testing
    bulk_root = ((fcc111, fcc111_root),
                 (bcc111, bcc111_root),
                 (hcp0001, hcp0001_root))

    for bulk, root_surf in bulk_root:
        prim = bulk("H", (1, 1, 2), a=1)

        # Check valid roots up to root 21 (the 10th root cell)
        assert fcc111_21_set == root_surface_analysis(prim, 21)

        # Use internal function
        internal_func_atoms = root_surface(prim, 7)

        # Remake using surface function
        helper_func_atoms = root_surf("H", 7, (1, 1, 2), a=1)

        # Right number of atoms
        assert len(internal_func_atoms) == 14
        assert len(helper_func_atoms) == 14
        assert (internal_func_atoms.cell == helper_func_atoms.cell).all()

    # Try bad root
    with pytest.raises(ValueError):
        fcc111_root("H", 5, (1, 1, 2), a=1)
