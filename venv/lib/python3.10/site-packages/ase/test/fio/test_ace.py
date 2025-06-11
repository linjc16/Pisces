import numpy as np
import pytest

import ase.units
from ase.atoms import Atoms
from ase.io.acemolecule import read_acemolecule_input, read_acemolecule_out


def test_acemolecule_output():

    sample_outfile = """\

====================  Atoms  =====================
 1       1.000000       2.000000      -0.6
 9       -1.000000       3.000000       0.7
==================================================

Total energy       = -1.5

!================================================
! Force:: List of total force in atomic unit.
! Atom           x         y         z
! Atom   0      0.1       0.2       0.3
! Atom   1      0.5       0.6       0.7
!================================================

    """
    with open('acemolecule_test.log', 'w') as fd:
        fd.write(sample_outfile)
    results = read_acemolecule_out('acemolecule_test.log')
    atoms = results['atoms']
    assert atoms.positions == pytest.approx(
        np.array([[1.0, 2.0, -0.6], [-1.0, 3.0, 0.7]]))
    assert all(atoms.symbols == 'HF')

    convert = ase.units.Hartree / ase.units.Bohr
    assert results['forces'] / convert == pytest.approx(
        np.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]]))
    assert results['energy'] / ase.units.Hartree == -1.5


def test_acemolecule_input():

    sample_inputfile = """\
%% BasicInformation
    Type Points
    Scaling 0.35
    Basis Sinc
    Grid Basic
    KineticMatrix Finite_Difference
    DerivativesOrder 7
    GeometryFilename acemolecule_test.xyz
    CellDimensionX 3.37316805
    CellDimensionY 3.37316805
    CellDimensionZ 3.37316805
    PointX 16
    PointY 16
    PointZ 16
    Periodicity 3
    %% Pseudopotential
        Pseudopotential 3
        PSFilePath PATH
        PSFileSuffix .PBE
    %% End
    GeometryFormat xyz
%% End
    """
    with open('acemolecule_test.inp', 'w') as fd:
        fd.write(sample_inputfile)
    atoms = Atoms(symbols='HF', positions=np.array(
        [[1.0, 2.0, -0.6], [-1.0, 3.0, 0.7]]))
    atoms.write('acemolecule_test.xyz', format='xyz')
    atoms = read_acemolecule_input('acemolecule_test.inp')
    assert atoms.positions == pytest.approx(
        np.array([[1.0, 2.0, -0.6], [-1.0, 3.0, 0.7]]))
    assert all(atoms.symbols == 'HF')
