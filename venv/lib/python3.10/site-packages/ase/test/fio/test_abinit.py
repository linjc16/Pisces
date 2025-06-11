from io import StringIO

import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.io import read, write
from ase.io.abinit import match_kpt_header, read_abinit_out, read_eig
from ase.units import Bohr, Hartree


def test_abinit_inputfile_roundtrip(testdir):
    m1 = bulk('Ti')
    m1.set_initial_magnetic_moments(range(len(m1)))
    write('abinit_save.in', images=m1, format='abinit-in')
    m2 = read('abinit_save.in', format='abinit-in')

    # (How many decimals?)
    assert not compare_atoms(m1, m2, tol=1e-7)


# "Hand-written" (reduced) abinit txt file based on v8.0.8 format:
sample_outfile = """\

.Version 8.0.8 of ABINIT

 -outvars: echo values of preprocessed input variables --------
            natom           2
           ntypat           1
            rprim      5.0  0.0  0.1
                       0.0  6.0  0.0
                       0.0  0.0  7.0
            typat      1  1
            znucl        8.0

================================

 ----iterations are completed or convergence reached----

 cartesian coordinates (angstrom) at end:
    1      2.5     2.5     3.7
    2      2.5     2.5     2.5

 cartesian forces (eV/Angstrom) at end:
    1     -0.1    -0.3    0.4
    2     -0.2    -0.4   -0.5

 Components of total free energy (in Hartree) :

    >>>>>>>>> Etotal= -42.5

 Cartesian components of stress tensor (hartree/bohr^3)
  sigma(1 1)=  2.3  sigma(3 2)=  3.1
  sigma(2 2)=  2.4  sigma(3 1)=  3.2
  sigma(3 3)=  2.5  sigma(2 1)=  3.3

END DATASET(S)
"""


def test_read_abinit_output():
    fd = StringIO(sample_outfile)
    results = read_abinit_out(fd)

    assert results.pop('version') == '8.0.8'

    atoms = results.pop('atoms')
    assert all(atoms.symbols == 'OO')
    assert atoms.positions == pytest.approx(
        np.array([[2.5, 2.5, 3.7], [2.5, 2.5, 2.5]]))
    assert all(atoms.pbc)
    assert atoms.cell[:] == pytest.approx(
        np.array([[5.0, 0.0, 0.1], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]]))

    ref_stress = pytest.approx([2.3, 2.4, 2.5, 3.1, 3.2, 3.3])
    assert results.pop('stress') / (Hartree / Bohr**3) == ref_stress
    assert results.pop('forces') == pytest.approx(
        np.array([[-0.1, -0.3, 0.4], [-0.2, -0.4, -0.5]]))

    for name in 'energy', 'free_energy':
        assert results.pop(name) / Hartree == -42.5

    assert not results


eig_text = """\
 Fermi (or HOMO) energy (hartree) =   0.123   Average Vxc (hartree)=  -0.456
 Eigenvalues (hartree) for nkpt=  2  k points:
 kpt#   1, nband=  3, wtk=  0.1, kpt=  0.2  0.3  0.4 (reduced coord)
  -0.2 0.2 0.3
 kpt#   2, nband=  3, wtk=  0.2, kpt=  0.3  0.4  0.5 (reduced coord)
  -0.3 0.4 0.5
"""


def test_parse_eig_with_fermiheader():
    eigval_ref = np.array([
        [-0.2, 0.2, 0.3],
        [-0.3, 0.4, 0.5]
    ]).reshape(1, 2, 3)  # spin x kpts x bands

    kpts_ref = np.array([
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5]
    ])

    weights_ref = [0.1, 0.2]

    eig_buf = StringIO(eig_text)
    data = read_eig(eig_buf)

    assert data['eigenvalues'] / Hartree == pytest.approx(eigval_ref)
    assert data['ibz_kpoints'] == pytest.approx(kpts_ref)
    assert data['kpoint_weights'] == pytest.approx(weights_ref)
    assert data['fermilevel'] / Hartree == pytest.approx(0.123)


def test_parse_eig_without_fermiheader():
    fd = StringIO(eig_text)
    next(fd)  # Header is omitted e.g. in non-selfconsistent calculations.

    data = read_eig(fd)
    assert 'fermilevel' not in data
    assert {'eigenvalues', 'ibz_kpoints', 'kpoint_weights'} == set(data)


def test_match_kpt_header():
    header_line = """\
kpt#  12, nband=  5, wtk=  0.02778, \
kpt=  0.4167  0.4167  0.0833 (reduced coord)
"""

    nbands, weight, vector = match_kpt_header(header_line)
    assert nbands == 5
    assert weight == pytest.approx(0.02778)
    assert vector == pytest.approx([0.4167, 0.4167, 0.0833])
