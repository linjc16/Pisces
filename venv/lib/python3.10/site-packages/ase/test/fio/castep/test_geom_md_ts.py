"""Tests for parsers of CASTEP .geom, .md, .ts files"""
import io

import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
from ase.io.castep import (
    read_castep_geom,
    read_castep_md,
    write_castep_geom,
    write_castep_md,
)


@pytest.fixture(name='images_ref')
def fixture_images():
    """Fixture of reference images"""
    atoms0 = bulk('Au', cubic=True)
    atoms0.rattle(seed=42)
    atoms1 = atoms0.copy()
    atoms1.symbols[0] = 'Cu'
    return [atoms0, atoms1]


def test_write_and_read_geom(images_ref):
    """Test if writing and reading .geom file get the original images back"""
    for atoms in images_ref:
        atoms.calc = EMT()
        atoms.get_stress()
    fd = io.StringIO()
    write_castep_geom(fd, images_ref)
    fd.seek(0)
    images = read_castep_geom(fd, index=':')
    assert len(images) == len(images_ref)
    for atoms, atoms_ref in zip(images, images_ref):
        assert not compare_atoms(atoms, atoms_ref)
        for key in ['free_energy', 'forces', 'stress']:
            np.testing.assert_allclose(
                atoms.calc.results[key],
                atoms_ref.calc.results[key],
                err_msg=key,
            )


def test_write_and_read_md(images_ref):
    """Test if writing and reading .md file get the original images back"""
    for atoms in images_ref:
        atoms.calc = EMT()
        atoms.get_stress()
    fd = io.StringIO()
    write_castep_md(fd, images_ref)
    fd.seek(0)
    images = read_castep_md(fd, index=':')
    assert len(images) == len(images_ref)
    for atoms, atoms_ref in zip(images, images_ref):
        assert not compare_atoms(atoms, atoms_ref)
        for key in ['free_energy', 'forces', 'stress']:
            np.testing.assert_allclose(
                atoms.calc.results[key],
                atoms_ref.calc.results[key],
                err_msg=key,
            )
