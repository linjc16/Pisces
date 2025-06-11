# additional tests of the dftb I/O
from io import StringIO

import numpy as np

from ase.atoms import Atoms
from ase.io.dftb import (
    read_dftb,
    read_dftb_lattice,
    read_dftb_velocities,
    write_dftb_velocities,
)
from ase.units import AUT, Bohr, second

# test ase.io.dftb.read_dftb
# with GenFormat-style Geometry section, periodic and non-periodic
fd_genformat_periodic = StringIO("""
Geometry = GenFormat {
4  S
O    C    H
1      1     -0.740273308080763      0.666649653991325      0.159416494587587
2      2      0.006891486298212     -0.006206095648781     -0.531735097642277
3      3      0.697047663527725      0.447111938577178     -1.264187748314973
4      3      0.036334158254826     -1.107555496919721     -0.464934648630337
0.000000000000000      0.000000000000000      0.000000000000000
3.750000000000000      0.000000000000000      0.000000000000000
1.500000000000000      4.500000000000000      0.000000000000000
0.450000000000000      1.050000000000000      3.750000000000000
}
Hamiltonian = DFTB {
}
Driver = {}
""")


fd_genformat_nonperiodic = StringIO("""
Geometry = GenFormat {
4  C
O    C    H
1      1     -0.740273308080763      0.666649653991325      0.159416494587587
2      2      0.006891486298212     -0.006206095648781     -0.531735097642277
3      3      0.697047663527725      0.447111938577178     -1.264187748314973
4      3      0.036334158254826     -1.107555496919721     -0.464934648630337
}
Hamiltonian = DFTB {
}
Driver = {}
""")


def test_read_dftb_genformat():
    positions = [[-0.740273308080763, 0.666649653991325, 0.159416494587587],
                 [0.006891486298212, -0.006206095648781, -0.531735097642277],
                 [0.697047663527725, 0.447111938577178, -1.264187748314973],
                 [0.036334158254826, -1.107555496919721, -0.464934648630337]]
    cell = [[3.75, 0., 0.], [1.5, 4.5, 0.], [0.45, 1.05, 3.75]]
    atoms = Atoms('OCH2', cell=cell, positions=positions)

    atoms.set_pbc(True)
    atoms_new = read_dftb(fd_genformat_periodic)
    assert np.all(atoms_new.numbers == atoms.numbers)
    assert np.allclose(atoms_new.positions, atoms.positions)
    assert np.all(atoms_new.pbc == atoms.pbc)
    assert np.allclose(atoms_new.cell, atoms.cell)

    atoms.set_pbc(False)
    atoms_new = read_dftb(fd_genformat_nonperiodic)
    assert np.all(atoms_new.numbers == atoms.numbers)
    assert np.allclose(atoms_new.positions, atoms.positions)
    assert np.all(atoms_new.pbc == atoms.pbc)
    assert np.allclose(atoms_new.cell, 0.)


# test ase.io.dftb.read_dftb (with explicit geometry specification;
# this GaAs geometry is borrowed from the DFTB+ v19.1 manual)
fd_explicit = StringIO("""
Geometry = {
  TypeNames = { "Ga" "As" }
  TypesAndCoordinates [Angstrom] = {
    1 0.000000   0.000000   0.000000
    2 1.356773   1.356773   1.356773
  }
  Periodic = Yes
  LatticeVectors [Angstrom] = {
    2.713546   2.713546   0.
    0.   2.713546   2.713546
    2.713546   0.   2.713546
  }
}
Hamiltonian = DFTB {
}
Driver = {}
""")


def test_read_dftb_explicit():
    x = 1.356773
    positions = [[0., 0., 0.], [x, x, x]]
    cell = [[2 * x, 2 * x, 0.], [0., 2 * x, 2 * x], [2 * x, 0., 2 * x]]
    atoms = Atoms('GaAs', cell=cell, positions=positions, pbc=True)

    atoms_new = read_dftb(fd_explicit)
    assert np.all(atoms_new.numbers == atoms.numbers)
    assert np.allclose(atoms_new.positions, atoms.positions)
    assert np.all(atoms_new.pbc == atoms.pbc)
    assert np.allclose(atoms_new.cell, atoms.cell)


# test ase.io.dftb.read_dftb_lattice
fd_lattice = StringIO("""
 MD step: 0
 Lattice vectors (A)
  26.1849388999576 5.773808884828536E-006 9.076696618724854E-006
 0.115834159141441 26.1947703089401 9.372892011565608E-006
 0.635711495837792 0.451552307731081 9.42069476334197
 Volume:                             0.436056E+05 au^3   0.646168E+04 A^3
 Pressure:                           0.523540E-04 au     0.154031E+10 Pa
 Gibbs free energy:               -374.4577147047 H       -10189.5129 eV
 Gibbs free energy including KE   -374.0819244147 H       -10179.2871 eV
 Potential Energy:                -374.4578629171 H       -10189.5169 eV
 MD Kinetic Energy:                  0.3757902900 H           10.2258 eV
 Total MD Energy:                 -374.0820726271 H       -10179.2911 eV
 MD Temperature:                     0.0009525736 au         300.7986 K
 MD step: 10
 Lattice vectors (A)
 26.1852379966047 5.130835479368833E-005 5.227350674663197E-005
 0.115884270570380 26.1953147133737 7.278784404810537E-005
 0.635711495837792 0.451552307731081 9.42069476334197
 Volume:                             0.436085E+05 au^3   0.646211E+04 A^3
 Pressure:                           0.281638E-04 au     0.828608E+09 Pa
 Gibbs free energy:               -374.5467030749 H       -10191.9344 eV
 Gibbs free energy including KE   -374.1009478784 H       -10179.8047 eV
 Potential Energy:                -374.5468512972 H       -10191.9384 eV
 MD Kinetic Energy:                  0.4457551965 H           12.1296 eV
 Total MD Energy:                 -374.1010961007 H       -10179.8088 eV
 MD Temperature:                     0.0011299245 au         356.8015 K
""")


def test_read_dftb_lattice():
    vectors = read_dftb_lattice(fd_lattice)
    mols = [Atoms(), Atoms()]
    read_dftb_lattice(fd_lattice, mols)

    compareVec = np.array([
        [26.1849388999576, 5.773808884828536e-6, 9.076696618724854e-6],
        [0.115834159141441, 26.1947703089401, 9.372892011565608e-6],
        [0.635711495837792, 0.451552307731081, 9.42069476334197]])

    assert (vectors[0] == compareVec).all()
    assert len(vectors) == 2
    assert len(vectors[1]) == 3
    assert (mols[0].get_cell() == compareVec).all()
    assert mols[1].get_pbc().all()


# test ase.io.dftb.read_dftb_velocities
geo_end_xyz = """
    2
MD iter: 0
    H    0.0    0.0  0.0  0.0     1.0   0.4  0.2
    H    0.0    0.0  0.0  0.0     0.8   1.4  2.0
    2
MD iter: 1
    H    0.0    0.0  0.0  0.0    -1.0  -0.4  0.2
    H    0.0    0.0  0.0  0.0     0.8   1.4  2.0
"""


def test_read_dftb_velocities():
    atoms = Atoms('H2')

    filename = 'geo_end.xyz'
    with open(filename, 'w') as fd:
        fd.write(geo_end_xyz)

    # Velocities (in Angstrom / ps) of the last MD iteration
    # The first 4 columns are the atom charge and coordinates
    read_dftb_velocities(atoms, filename=filename)

    velocities = np.linspace(-1, 2, num=6).reshape(2, 3)
    velocities /= 1e-12 * second
    assert np.allclose(velocities, atoms.get_velocities())


# test ase.io.dftb.write_dftb_velocities
def test_write_dftb_velocities():
    atoms = Atoms('H2')

    velocities = np.linspace(-1, 2, num=6).reshape(2, 3)
    atoms.set_velocities(velocities)

    write_dftb_velocities(atoms, filename='velocities.txt')

    velocities = np.loadtxt('velocities.txt') * Bohr / AUT
    assert np.allclose(velocities, atoms.get_velocities())
