"""Test write_input"""
import numpy as np
import pytest

from ase import Atoms
from ase.calculators.siesta.parameters import PAOBasisBlock, Species


@pytest.fixture(name="atoms_h")
def fixture_atoms_h():
    """hydrogen atom"""
    return Atoms('H', [(0.0, 0.0, 0.0)])


@pytest.fixture(name="atoms_ch4")
def fixture_atoms_ch4():
    """methane molecule"""
    positions = [
        [0.000000, 0.000000, 0.000000],
        [0.682793, 0.682793, 0.682793],
        [-0.682793, -0.682793, 0.682790],
        [-0.682793, 0.682793, -0.682793],
        [0.682793, -0.682793, -0.682793],
    ]
    return Atoms('CH4', positions)


@pytest.mark.calculator_lite()
@pytest.mark.calculator('siesta')
def test_simple(factory, atoms_h):
    """Test simple fdf-argument case."""
    siesta = factory.calc(
        label='test_label',
        fdf_arguments={'DM.Tolerance': 1e-3})
    atoms_h.calc = siesta
    siesta.write_input(atoms_h, properties=['energy'])
    with open('test_label.fdf', encoding='utf-8') as fd:
        lines = fd.readlines()
    assert any(line.split() == ['DM.Tolerance', '0.001'] for line in lines)


@pytest.mark.calculator_lite()
@pytest.mark.calculator('siesta')
def test_complex(factory, atoms_h):
    """Test (slightly) more complex case of setting fdf-arguments."""
    siesta = factory.calc(
        label='test_label',
        mesh_cutoff=3000,
        fdf_arguments={
            'DM.Tolerance': 1e-3,
            'ON.eta': (5, 'Ry')})
    atoms_h.calc = siesta
    siesta.write_input(atoms_h, properties=['energy'])
    with open('test_label.fdf', encoding='utf-8') as fd:
        lines = fd.readlines()

    assert 'MeshCutoff\t3000\teV\n' in lines
    assert 'DM.Tolerance\t0.001\n' in lines
    assert 'ON.eta\t5\tRy\n' in lines


@pytest.mark.calculator_lite()
@pytest.mark.calculator('siesta')
def test_set_fdf_arguments(factory, atoms_h):
    """Test setting fdf-arguments after initiation."""
    siesta = factory.calc(
        label='test_label',
        mesh_cutoff=3000,
        fdf_arguments={
            'DM.Tolerance': 1e-3,
            'ON.eta': (5, 'Ry')})
    siesta.set_fdf_arguments(
        {'DM.Tolerance': 1e-2,
         'ON.eta': (2, 'Ry')})
    siesta.write_input(atoms_h, properties=['energy'])
    with open('test_label.fdf', encoding='utf-8') as fd:
        lines = fd.readlines()
    assert 'MeshCutoff\t3000\teV\n' in lines
    assert 'DM.Tolerance\t0.01\n' in lines
    assert 'ON.eta\t2\tRy\n' in lines


@pytest.mark.calculator_lite()
@pytest.mark.calculator('siesta')
def test_species(factory, atoms_ch4):
    """Test initiation using Species."""
    siesta = factory.calc()
    species, numbers = siesta.species(atoms_ch4)
    assert all(numbers == np.array([1, 2, 2, 2, 2]))

    siesta = factory.calc(species=[Species(symbol='C', tag=1)])
    species, numbers = siesta.species(atoms_ch4)
    assert all(numbers == np.array([1, 2, 2, 2, 2]))

    atoms_ch4.set_tags([0, 0, 0, 1, 0])
    species, numbers = siesta.species(atoms_ch4)
    assert all(numbers == np.array([1, 2, 2, 2, 2]))

    siesta = factory.calc(
        species=[
            Species(symbol='C', tag=-1),
            Species(
                symbol='H',
                tag=1,
                basis_set='SZ',
                pseudopotential='somepseudo')])

    species, numbers = siesta.species(atoms_ch4)
    assert all(numbers == np.array([1, 2, 2, 4, 2]))

    siesta = factory.calc(label='test_label', species=species)
    siesta.write_input(atoms_ch4, properties=['energy'])
    with open('test_label.fdf', encoding='utf-8') as fd:
        lines = fd.readlines()

    lines = [line.split() for line in lines]
    assert ['1', '6', 'C.lda.1'] in lines
    assert ['2', '1', 'H.lda.2'] in lines
    assert ['4', '1', 'H.4', 'H.4.psml'] in lines
    assert ['C.lda.1', 'DZP'] in lines
    assert ['H.lda.2', 'DZP'] in lines
    assert ['H.4', 'SZ'] in lines


@pytest.mark.calculator_lite()
@pytest.mark.calculator('siesta')
def test_pao_block(factory, atoms_ch4):
    """Test if PAO block can be given as species."""
    c_basis = """2 nodes 1.00
    0 1 S 0.20 P 1 0.20 6.00
    5.00
    1.00
    1 2 S 0.20 P 1 E 0.20 6.00
    6.00 5.00
    1.00 0.95"""
    basis_set = PAOBasisBlock(c_basis)
    species = Species(symbol='C', basis_set=basis_set)
    siesta = factory.calc(label='test_label', species=[species])
    siesta.write_input(atoms_ch4, properties=['energy'])
    with open('test_label.fdf', encoding='utf-8') as fd:
        lines = fd.readlines()
    lines = [line.split() for line in lines]
    assert ['%block', 'PAO.Basis'] in lines
    assert ['%endblock', 'PAO.Basis'] in lines
