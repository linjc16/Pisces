import io

import numpy as np

from ase import Atoms

# from ase.io import read
from ase.calculators.openmx.reader import read_eigenvalues, read_openmx
from ase.units import Bohr, Ha

openmx_out_sample = """
System.CurrentDirectory        ./
System.Name        ch4

Atoms.SpeciesAndCoordinates.Unit        Ang
<Definition.of.Atomic.Species
    C  C5.0-s1p1  C_PBE19
    H  H5.0-s1  H_PBE19
Definition.of.Atomic.Species>

Atoms.SpeciesAndCoordinates.Unit        Ang
<Atoms.SpeciesAndCoordinates
    1  C  0.0  0.0  0.1  2.0  2.0
    2  H  0.682793  0.682793  0.682793  0.5  0.5
    3  H  -0.682793  -0.682793  0.68279  0.5  0.5
    4  H  -0.682793  0.682793  -0.682793  0.5  0.5
    5  H  0.682793  -0.682793  -0.682793  0.5  0.5
Atoms.SpeciesAndCoordinates>

<Atoms.UnitVectors
    10.0  0.0  0.0
    0.0  10.0  0.0
    0.0  0.0  10.0
Atoms.UnitVectors>

scf.EigenvalueSolver        Band

  ...

  Utot.         -8.055096450113

  ...

  Chemical potential (Hartree)      -0.156250000000

  ...

*************************************************************************
*************************************************************************
            Decomposed energies in Hartree unit

   Utot = Utot(up) + Utot(dn)
        = Ukin(up) + Ukin(dn) + Uv(up) + Uv(dn)
        + Ucon(up)+ Ucon(dn) + Ucore+UH0 + Uvdw

   Uele = Ukin(up) + Ukin(dn) + Uv(up) + Uv(dn)
   Ucon arizes from a constant potential added in the formalism

           up: up spin state, dn: down spin state
*************************************************************************
*************************************************************************

  Total energy (Hartree) = -8.055096425922011

  Decomposed.energies.(Hartree).with.respect.to.atom

                 Utot
     1    C     -6.261242355014   ...
     2    H     -0.445956460556   ...
     3    H     -0.445956145906   ...
     4    H     -0.450970732231   ...
     5    H     -0.450970732215   ...

  ...

<coordinates.forces
  5
    1     C     0.00   0.00   0.10   0.00000  0.00000 -0.091659
    2     H     0.68   0.68   0.68   0.02700  0.02700  0.029454
    3     H    -0.68  -0.68   0.68  -0.02700 -0.02700  0.029455
    4     H    -0.68   0.68  -0.68   0.00894 -0.00894  0.016362
    5     H     0.68  -0.68  -0.68  -0.00894  0.00894  0.016362
coordinates.forces>

  ...

***********************************************************
***********************************************************
       Fractional coordinates of the final structure
***********************************************************
***********************************************************

     1      C     0.00000   0.00000   0.01000
     2      H     0.06827   0.06827   0.06827
     3      H     0.93172   0.93172   0.06827
     4      H     0.93172   0.06827   0.93172
     5      H     0.06827   0.93172   0.93172

...

"""


def test_openmx_out(config_file):
    with open('openmx_fio_test.out', 'w') as fd:
        fd.write(openmx_out_sample)
    atoms = read_openmx('openmx_fio_test')
    tol = 1e-2

    # Expected values
    energy = -8.0551
    energies = np.array([-6.2612, -0.4459, -0.4459, -0.4509, -0.4509])
    forces = np.array([[0.00000, 0.00000, -0.091659],
                       [0.02700, 0.02700, 0.029454],
                       [-0.02700, -0.02700, 0.029455],
                       [0.00894, -0.00894, 0.016362],
                       [-0.00894, 0.00894, 0.016362]])

    assert isinstance(atoms, Atoms)

    assert np.isclose(atoms.calc.results['energy'], energy * Ha, atol=tol)
    assert np.all(np.isclose(atoms.calc.results['energies'],
                  energies * Ha, atol=tol))
    assert np.all(np.isclose(atoms.calc.results['forces'],
                  forces * Ha / Bohr, atol=tol))


openmx_eigenvalues_gamma_sample = """
...

***********************************************************
***********************************************************
            Eigenvalues (Hartree) for SCF KS-eq.
***********************************************************
***********************************************************

   Chemical Potential (Hartree) =  -0.12277513509616
   Number of States             =  58.00000000000000
   HOMO = 29
   Eigenvalues
                Up-spin            Down-spin
          1  -0.96233478518931  -0.96233478518931
          2  -0.94189339856450  -0.94189339856450
          3  -0.86350555424836  -0.86350555424836
          4  -0.83918201748919  -0.83918201748919
          5  -0.72288697309928  -0.72288697309928
          6  -0.67210805969879  -0.67210805969879
          7  -0.64903406048465  -0.64903406048465
          8  -0.58249976216367  -0.58249976216367
          9  -0.55161386332358  -0.55161386332358

***********************************************************
***********************************************************
              History of cell optimization
***********************************************************
***********************************************************

...

"""

openmx_eigenvalues_bulk_sample = """
...

scf.Kgrid = 2 1 1

...

***********************************************************
***********************************************************
           Eigenvalues (Hartree) of SCF KS-eq.
***********************************************************
***********************************************************

   Chemical Potential (Hatree) =  -0.19810093996855
   Number of States            = 156.00000000000000
   Eigenvalues
              Up-spin           Down-spin

   kloop=0
   k1=  -0.44444 k2=  -0.44445 k3=   0.00000

    1   -2.33424746491277  -2.33424746917880
    2   -2.33424055817432  -2.33424056243807
    3   -2.33419668076232  -2.33419668261398
    4   -1.46440634271635  -1.46440634435648
    5   -1.46439286017722  -1.46439286180118
    6   -1.46436086583111  -1.46436086399066
    7   -1.46397017044962  -1.46397017874325
    8   -1.46394407220255  -1.46394408049882
    9   -1.46389030794971  -1.46389031384386

   kloop=1
   k1=  -0.44444 k2=  -0.33333 k3=   0.00000

    1   -2.33424705259020  -2.33424705685571
    2   -2.33424133604313  -2.33424134030309
    3   -2.33419651862703  -2.33419652048304
    4   -1.46440529840756  -1.46440530004421
    5   -1.46439446518585  -1.46439446677862
    6   -1.46436032862668  -1.46436032682027
    7   -1.46396740984959  -1.46396741813205
    8   -1.46394638210900  -1.46394639039694
    9   -1.46389029838585  -1.46389030429995

***********************************************************
***********************************************************
              History of geometry optimization
***********************************************************
***********************************************************
...
"""


def test_openmx_read_eigenvalues():
    tol = 1e-2
    # reader.py -> `def read_file(filename...)` -> patterns
    eigenvalues_pattern = "Eigenvalues (Hartree)"
    with io.StringIO(openmx_eigenvalues_gamma_sample) as fd:
        while True:
            line = fd.readline()
            if eigenvalues_pattern in line:
                break
        eigenvalues = read_eigenvalues(line, fd)

    gamma_eigenvalues = np.array([[[-0.96233478518931, -0.96233478518931],
                                  [-0.94189339856450, -0.94189339856450],
                                  [-0.86350555424836, -0.86350555424836],
                                  [-0.83918201748919, -0.83918201748919],
                                  [-0.72288697309928, -0.72288697309928],
                                  [-0.67210805969879, -0.67210805969879],
                                  [-0.64903406048465, -0.64903406048465],
                                  [-0.58249976216367, -0.58249976216367],
                                  [-0.55161386332358, -0.55161386332358]]])
    gamma_eigenvalues = np.swapaxes(gamma_eigenvalues.T, 1, 2)

    assert np.all(np.isclose(eigenvalues, gamma_eigenvalues, atol=tol))

    with io.StringIO(openmx_eigenvalues_bulk_sample) as fd:
        while True:
            line = fd.readline()
            if eigenvalues_pattern in line:
                break
        eigenvalues = read_eigenvalues(line, fd)

    bulk_eigenvalues = np.array([[[-2.33424746491277, -2.33424746917880],
                                 [-2.33424055817432, -2.33424056243807],
                                 [-2.33419668076232, -2.33419668261398],
                                 [-1.46440634271635, -1.46440634435648],
                                 [-1.46439286017722, -1.46439286180118],
                                 [-1.46436086583111, -1.46436086399066],
                                 [-1.46397017044962, -1.46397017874325],
                                 [-1.46394407220255, -1.46394408049882],
                                 [-1.46389030794971, -1.46389031384386]],
                                 [[-2.33424705259020, -2.33424705685571],
                                 [-2.33424133604313, -2.33424134030309],
                                 [-2.33419651862703, -2.33419652048304],
                                 [-1.46440529840756, -1.46440530004421],
                                 [-1.46439446518585, -1.46439446677862],
                                 [-1.46436032862668, -1.46436032682027],
                                 [-1.46396740984959, -1.46396741813205],
                                 [-1.46394638210900, -1.46394639039694],
                                 [-1.46389029838585, -1.46389030429995]]])
    bulk_eigenvalues = np.swapaxes(bulk_eigenvalues.T, 1, 2)

    assert np.all(np.isclose(eigenvalues[:, :2, :], bulk_eigenvalues, atol=tol))
