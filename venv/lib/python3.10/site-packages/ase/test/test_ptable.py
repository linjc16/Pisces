#!/usr/bin/env python3
"""Test Periodic Table."""
from ase.utils.ptable import ptable

spacing = 1.0


def test_pourbaix():
    """Test the periodic table generating function."""
    atoms = ptable(spacing)
    x = atoms.positions[:, 0]
    # testing to ensure that the table is 18 groups wide
    assert spacing * 17 == (x.max() - x.min())
    # testing that there are 118 unique elements
    symbols = set(atoms.get_chemical_symbols())
    assert len(symbols) == 118
