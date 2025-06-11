from io import StringIO

import numpy as np

from ase.io.espresso import (
    Namelist,
    read_espresso_ph,
    read_fortran_namelist,
    write_espresso_ph,
)


def test_write_espresso_ph_single():
    input_data = {
        "amass(1)": 1.0,
        "amass(2)": 2.0,
        "prefix": "prefix",
        "outdir": "/path/to/outdir",
        "eth_rps": 0.1,
        "qplot": False,
        "ldisp": False,
        "trans": True,
        "tr2_ph": 1e-12,
        "alpha_mix(1)": 0.1,
        "nat_todo": 0,
    }

    qpts = (0.5, -0.1, 1 / 3)

    input_data = Namelist(input_data)
    input_data.to_nested('ph')

    string_io = StringIO()

    write_espresso_ph(string_io, input_data=input_data, qpts=qpts)

    expected = (
        "&INPUTPH\n"
        "   amass(1)         = 1.0\n"
        "   amass(2)         = 2.0\n"
        "   outdir           = '/path/to/outdir'\n"
        "   prefix           = 'prefix'\n"
        "   tr2_ph           = 1e-12\n"
        "   alpha_mix(1)     = 0.1\n"
        "   trans            = .true.\n"
        "   eth_rps          = 0.1\n"
        "   qplot            = .false.\n"
        "   ldisp            = .false.\n"
        "   nat_todo         = 0\n"
        "/\n"
        "0.50000000 -0.10000000 0.33333333\n"
    )

    string_io.seek(0)

    recycled_input_data = read_fortran_namelist(string_io)[0]

    assert recycled_input_data == input_data
    assert string_io.getvalue() == expected


def test_write_espresso_ph_list():
    input_data = {
        "amass(1)": 1.0,
        "amass(2)": 2.0,
        "prefix": "prefix",
        "outdir": "/path/to/outdir",
        "eth_rps": 0.1,
        "qplot": True,
        "ldisp": True,
    }

    qpts = [(0.5, -0.1, 1 / 3, 2), (0.1, 0.2, 0.3, 10), (0.2, 0.3, 0.4, 1)]

    string_io = StringIO()

    input_data = Namelist(input_data)
    input_data.to_nested('ph')

    write_espresso_ph(string_io, input_data=input_data, qpts=qpts)

    expected = (
        "&INPUTPH\n"
        "   amass(1)         = 1.0\n"
        "   amass(2)         = 2.0\n"
        "   outdir           = '/path/to/outdir'\n"
        "   prefix           = 'prefix'\n"
        "   eth_rps          = 0.1\n"
        "   qplot            = .true.\n"
        "   ldisp            = .true.\n"
        "/\n"
        "3\n"
        "0.50000000 -0.10000000 0.33333333 2\n"
        "0.10000000 0.20000000 0.30000000 10\n"
        "0.20000000 0.30000000 0.40000000 1\n"
    )

    string_io.seek(0)

    recycled_input_data = read_fortran_namelist(string_io)[0]

    assert recycled_input_data == input_data
    assert string_io.getvalue() == expected


def test_write_espresso_ph_nat_todo():
    input_data = {
        "amass(1)": 1.0,
        "amass(2)": 2.0,
        "prefix": "prefix",
        "outdir": "/path/to/outdir",
        "eth_rps": 0.1,
        "qplot": True,
        "nat_todo": 3,
        "ldisp": True,
    }

    qpts = [(0.5, -0.1, 1 / 3, 1), (0.1, 0.2, 0.3, -1), (0.2, 0.3, 0.4, 4)]

    input_data = Namelist(input_data)
    input_data.to_nested('ph')

    string_io = StringIO()

    write_espresso_ph(
        string_io,
        input_data=input_data,
        qpts=qpts,
        nat_todo_indices=[
            1,
            2,
            3])

    expected = (
        "&INPUTPH\n"
        "   amass(1)         = 1.0\n"
        "   amass(2)         = 2.0\n"
        "   outdir           = '/path/to/outdir'\n"
        "   prefix           = 'prefix'\n"
        "   eth_rps          = 0.1\n"
        "   qplot            = .true.\n"
        "   ldisp            = .true.\n"
        "   nat_todo         = 3\n"
        "/\n"
        "3\n"
        "0.50000000 -0.10000000 0.33333333 1\n"
        "0.10000000 0.20000000 0.30000000 -1\n"
        "0.20000000 0.30000000 0.40000000 4\n"
        "1 2 3\n"
    )

    string_io.seek(0)

    recycled_input_data = read_fortran_namelist(string_io)[0]

    assert recycled_input_data == input_data
    assert string_io.getvalue() == expected


simple_ph_output = """
     Program PHONON v.6.0 (svn rev. 13188M) starts on  7Dec2016 at 13:16:14

     Calculation of q =    0.0000000   0.0000000   1.0000000

     celldm(1)=   6.650000  celldm(2)=   0.000000  celldm(3)=   0.000000
     celldm(4)=   0.000000  celldm(5)=   0.000000  celldm(6)=   0.000000

     crystal axes: (cart. coord. in units of alat)
               a(1) = (  -0.500000   0.000000   0.500000 )
               a(2) = (   0.000000   0.500000   0.500000 )
               a(3) = (  -0.500000   0.500000   0.000000 )

   Cartesian axes

     site n.     atom                  positions (alat units)
         1           Ni  tau(   1) = (   0.0000000   0.0000000   0.0000000  )

     number of k points=   216  Marzari-Vanderbilt smearing, width (Ry)=  0.0200

     Number of k-points >= 100: set verbosity='high' to print them.

     Number of k-points >= 100: set verbosity='high' to print the bands.

     the Fermi energy is    14.2603 ev

     celldm(1)=    6.65000  celldm(2)=    0.00000  celldm(3)=    0.00000
     celldm(4)=    0.00000  celldm(5)=    0.00000  celldm(6)=    0.00000

     crystal axes: (cart. coord. in units of alat)
               a(1) = ( -0.5000  0.0000  0.5000 )
               a(2) = (  0.0000  0.5000  0.5000 )
               a(3) = ( -0.5000  0.5000  0.0000 )

     Atoms inside the unit cell:

     Cartesian axes

     site n.  atom      mass           positions (alat units)
        1     Ni  58.6934   tau(    1) = (    0.00000    0.00000    0.00000  )

     Computing dynamical matrix for
                    q = (   0.0000000   0.0000000   1.0000000 )

     number of k points=   216  Marzari-Vanderbilt smearing, width (Ry)=  0.0200


     Mode symmetry, D_2h (mmm)  point group:


     Atomic displacements:
     There are   3 irreducible representations

     Representation     1      1 modes -B_1u  To be done

     Representation     2      1 modes -B_2u  To be done

     Representation     3      1 modes -B_3u  To be done



     Alpha used in Ewald sum =   2.8000
     PHONON       :    18.15s CPU        23.25s WALL

     Number of q in the star =    2
     List of q in the star:
          1   0.000000000   0.000000000   1.000000000
          2   1.000000000   0.000000000   0.000000000

     Diagonalizing the dynamical matrix

     q = (    0.000000000   0.000000000   1.000000000 )

 **************************************************************************
     freq (    1) =       6.382312 [THz] =     212.891027 [cm-1]
     freq (    2) =       6.382357 [THz] =     212.892523 [cm-1]
     freq (    3) =       8.906622 [THz] =     297.092916 [cm-1]
 **************************************************************************

     Mode symmetry, D_2h (mmm)  [C_2h (2/m) ] magnetic point group:

     freq (  1 -  1) =        212.9  [cm-1]   --> B_2u
     freq (  2 -  2) =        212.9  [cm-1]   --> B_3u
     freq (  3 -  3) =        297.1  [cm-1]   --> B_1u

     init_run     :      0.24s CPU      0.35s WALL (       1 calls)"""


def test_read_espresso_ph_all():
    fd = StringIO(simple_ph_output)
    read_espresso_ph(fd)


complex_ph_output = """
     Program PHONON v.6.0 (svn rev. 13188M) starts on  7Dec2016 at 10:43: 7


     Dynamical matrices for ( 4, 4, 4)  uniform grid of q-points
     (   8q-points):
       N         xq(1)         xq(2)         xq(3)
       1   0.000000000   0.000000000   0.000000000
       2  -0.250000000   0.250000000  -0.250000000
       3   0.500000000  -0.500000000   0.500000000
       4   0.000000000   0.500000000   0.000000000
       5   0.750000000  -0.250000000   0.750000000
       6   0.500000000   0.000000000   0.500000000
       7   0.000000000  -1.000000000   0.000000000
       8  -0.500000000  -1.000000000   0.000000000

     Calculation of q =    0.0000000   0.0000000   0.0000000

     Electron-phonon coefficients for Al

     bravais-lattice index     =            2
     lattice parameter (alat)  =       7.5000  a.u.
     unit-cell volume          =     105.4688 (a.u.)^3
     number of atoms/cell      =            1
     number of atomic types    =            1
     kinetic-energy cut-off    =      15.0000  Ry
     charge density cut-off    =      60.0000  Ry
     convergence threshold     =      1.0E-10
     beta                      =       0.7000
     number of iterations used =            4
     Exchange-correlation      =  SLA  PZ   NOGX NOGC ( 1  1  0  0 0 0)


     celldm(1)=    7.50000  celldm(2)=    0.00000  celldm(3)=    0.00000
     celldm(4)=    0.00000  celldm(5)=    0.00000  celldm(6)=    0.00000

     crystal axes: (cart. coord. in units of alat)
               a(1) = ( -0.5000  0.0000  0.5000 )
               a(2) = (  0.0000  0.5000  0.5000 )
               a(3) = ( -0.5000  0.5000  0.0000 )

     reciprocal axes: (cart. coord. in units 2 pi/alat)
               b(1) = ( -1.0000 -1.0000  1.0000 )
               b(2) = (  1.0000  1.0000  1.0000 )
               b(3) = ( -1.0000  1.0000 -1.0000 )


     Atoms inside the unit cell:

     Cartesian axes

     site n.  atom      mass           positions (alat units)
        1     Al  26.9800   tau(    1) = (    0.00000    0.00000    0.00000  )

     Computing dynamical matrix for
                    q = (   0.0000000   0.0000000   0.0000000 )

     49 Sym.Ops. (with q -> -q+G )

     Mode symmetry, O_h (m-3m)  point group:


     Atomic displacements:
     There are   1 irreducible representations

     Representation     1      3 modes -T_1u G_15  G_4-  To be done



     Alpha used in Ewald sum =   0.7000
     PHONON       :     0.17s CPU         0.19s WALL



     Representation #  1 modes #   1  2  3

     Number of q in the star =    1
     List of q in the star:
          1   0.000000000   0.000000000   0.000000000

     Diagonalizing the dynamical matrix

     q = (    0.000000000   0.000000000   0.000000000 )

 **************************************************************************
     freq (    1) =       0.173268 [THz] =       5.779601 [cm-1]
     freq (    2) =       0.173268 [THz] =       5.779601 [cm-1]
     freq (    3) =       0.173268 [THz] =       5.779601 [cm-1]
 **************************************************************************

     Mode symmetry, O_h (m-3m)  point group:

     freq (  1 -  3) =          5.8  [cm-1]   --> T_1u G_15  G_4- I
     electron-phonon interaction  ...

     Gaussian Broadening:   0.005 Ry, ngauss=   0
     DOS =  1.339210 states/spin/Ry/Unit Cell at Ef=  8.321793 eV
     lambda(    1)=  0.0000   gamma=    0.00 GHz
     lambda(    2)=  0.0000   gamma=    0.00 GHz
     lambda(    3)=  0.0000   gamma=    0.00 GHz
     Gaussian Broadening:   0.010 Ry, ngauss=   0
     DOS =  1.881761 states/spin/Ry/Unit Cell at Ef=  8.327153 eV
     lambda(    1)=  0.0000   gamma=    0.00 GHz
     lambda(    2)=  0.0000   gamma=    0.00 GHz
     lambda(    3)=  0.0000   gamma=    0.00 GHz
     Gaussian Broadening:   0.015 Ry, ngauss=   0
     DOS =  2.123229 states/spin/Ry/Unit Cell at Ef=  8.328621 eV
     lambda(    1)=  0.0000   gamma=    0.00 GHz
     lambda(    2)=  0.0000   gamma=    0.00 GHz
     lambda(    3)=  0.0000   gamma=    0.00 GHz
     Gaussian Broadening:   0.020 Ry, ngauss=   0
     DOS =  2.249739 states/spin/Ry/Unit Cell at Ef=  8.324319 eV
     lambda(    1)=  0.0000   gamma=    0.02 GHz
     lambda(    2)=  0.0000   gamma=    0.03 GHz
     lambda(    3)=  0.0000   gamma=    0.03 GHz
     Gaussian Broadening:   0.025 Ry, ngauss=   0
     DOS =  2.329803 states/spin/Ry/Unit Cell at Ef=  8.317861 eV
     lambda(    1)=  0.0000   gamma=    0.08 GHz
     lambda(    2)=  0.0000   gamma=    0.09 GHz
     lambda(    3)=  0.0000   gamma=    0.09 GHz
     Gaussian Broadening:   0.030 Ry, ngauss=   0
     DOS =  2.396029 states/spin/Ry/Unit Cell at Ef=  8.311296 eV
     lambda(    1)=  0.0000   gamma=    0.16 GHz
     lambda(    2)=  0.0000   gamma=    0.18 GHz
     lambda(    3)=  0.0000   gamma=    0.18 GHz
     Gaussian Broadening:   0.035 Ry, ngauss=   0
     DOS =  2.455226 states/spin/Ry/Unit Cell at Ef=  8.305262 eV
     lambda(    1)=  0.0000   gamma=    0.25 GHz
     lambda(    2)=  0.0000   gamma=    0.27 GHz
     lambda(    3)=  0.0000   gamma=    0.27 GHz
     Gaussian Broadening:   0.040 Ry, ngauss=   0
     DOS =  2.507873 states/spin/Ry/Unit Cell at Ef=  8.299955 eV
     lambda(    1)=  0.0000   gamma=    0.35 GHz
     lambda(    2)=  0.0000   gamma=    0.38 GHz
     lambda(    3)=  0.0000   gamma=    0.38 GHz
     Gaussian Broadening:   0.045 Ry, ngauss=   0
     DOS =  2.552966 states/spin/Ry/Unit Cell at Ef=  8.295411 eV
     lambda(    1)=  0.0000   gamma=    0.48 GHz
     lambda(    2)=  0.0000   gamma=    0.50 GHz
     lambda(    3)=  0.0000   gamma=    0.50 GHz
     Gaussian Broadening:   0.050 Ry, ngauss=   0
     DOS =  2.589582 states/spin/Ry/Unit Cell at Ef=  8.291553 eV
     lambda(    1)=  0.0000   gamma=    0.61 GHz
     lambda(    2)=  0.0000   gamma=    0.63 GHz
     lambda(    3)=  0.0000   gamma=    0.64 GHz


     Number of q in the star =    1
     List of q in the star:
          1   0.000000000   0.000000000   0.000000000

     Calculation of q =   -0.2500000   0.2500000  -0.2500000

     PseudoPot. # 1 for Al read from file:
     ./Al.pz-vbc.UPF
     MD5 check sum: 614279c88ff8d45c90147292d03ed420
     Pseudo is Norm-conserving, Zval =  3.0
     Generated by new atomic code, or converted to UPF format
     Using radial grid of  171 points,  2 beta functions with:
                l(1) =   0
                l(2) =   1

     atomic species   valence    mass     pseudopotential
        Al             3.00    26.98000     Al( 1.00)

     48 Sym. Ops., with inversion, found

     bravais-lattice index     =            2
     lattice parameter (alat)  =       7.5000  a.u.
     unit-cell volume          =     105.4688 (a.u.)^3
     number of atoms/cell      =            1
     number of atomic types    =            1
     kinetic-energy cut-off    =      15.0000  Ry
     charge density cut-off    =      60.0000  Ry
     convergence threshold     =      1.0E-10
     beta                      =       0.7000
     number of iterations used =            4
     Exchange-correlation      =  SLA  PZ   NOGX NOGC ( 1  1  0  0 0 0)


     celldm(1)=    7.50000  celldm(2)=    0.00000  celldm(3)=    0.00000
     celldm(4)=    0.00000  celldm(5)=    0.00000  celldm(6)=    0.00000

     crystal axes: (cart. coord. in units of alat)
               a(1) = ( -0.5000  0.0000  0.5000 )
               a(2) = (  0.0000  0.5000  0.5000 )
               a(3) = ( -0.5000  0.5000  0.0000 )

     reciprocal axes: (cart. coord. in units 2 pi/alat)
               b(1) = ( -1.0000 -1.0000  1.0000 )
               b(2) = (  1.0000  1.0000  1.0000 )
               b(3) = ( -1.0000  1.0000 -1.0000 )


     Atoms inside the unit cell:

     Cartesian axes

     site n.  atom      mass           positions (alat units)
        1     Al  26.9800   tau(    1) = (    0.00000    0.00000    0.00000  )

     Computing dynamical matrix for
                    q = (  -0.2500000   0.2500000  -0.2500000 )

      6 Sym.Ops. (no q -> -q+G )


     G cutoff =   85.4897  (    218 G-vectors)     FFT grid: ( 15, 15, 15)

     number of k points=   240  Marzari-Vanderbilt smearing, width (Ry)=  0.0500

     PseudoPot. # 1 for Al read from file:
     ./Al.pz-vbc.UPF
     MD5 check sum: 614279c88ff8d45c90147292d03ed420
     Pseudo is Norm-conserving, Zval =  3.0
     Generated by new atomic code, or converted to UPF format
     Using radial grid of  171 points,  2 beta functions with:
                l(1) =   0
                l(2) =   1

     Mode symmetry, C_3v (3m)   point group:


     Atomic displacements:
     There are   2 irreducible representations

     Representation     1      1 modes -A_1  L_1  To be done

     Representation     2      2 modes -E    L_3  To be done



     Alpha used in Ewald sum =   0.7000
     PHONON       :     3.67s CPU         3.94s WALL



     Representation #  1 mode #   1

     Self-consistent Calculation

      iter #   1 total cpu time :     4.0 secs   av.it.:   4.2
      thresh= 1.000E-02 alpha_mix =  0.700 |ddv_scf|^2 =  2.094E-02

      iter #   2 total cpu time :     4.1 secs   av.it.:   4.9
      thresh= 1.000E-02 alpha_mix =  0.700 |ddv_scf|^2 =  9.107E-01

      iter #   3 total cpu time :     4.2 secs   av.it.:   4.8
      thresh= 1.000E-02 alpha_mix =  0.700 |ddv_scf|^2 =  5.162E-07

      iter #   4 total cpu time :     4.3 secs   av.it.:   5.2
      thresh= 7.185E-05 alpha_mix =  0.700 |ddv_scf|^2 =  2.353E-09

      iter #   5 total cpu time :     4.4 secs   av.it.:   5.4
      thresh= 4.851E-06 alpha_mix =  0.700 |ddv_scf|^2 =  1.600E-10

      iter #   6 total cpu time :     4.5 secs   av.it.:   5.2
      thresh= 1.265E-06 alpha_mix =  0.700 |ddv_scf|^2 =  9.187E-11

     End of self-consistent calculation

     Convergence has been achieved


     Representation #  2 modes #   2  3

     Self-consistent Calculation

      iter #   1 total cpu time :     4.7 secs   av.it.:   3.5
      thresh= 1.000E-02 alpha_mix =  0.700 |ddv_scf|^2 =  3.275E-08

      iter #   2 total cpu time :     4.9 secs   av.it.:   6.0
      thresh= 1.810E-05 alpha_mix =  0.700 |ddv_scf|^2 =  3.070E-09

      iter #   3 total cpu time :     5.1 secs   av.it.:   5.7
      thresh= 5.541E-06 alpha_mix =  0.700 |ddv_scf|^2 =  1.011E-11

     End of self-consistent calculation

     Convergence has been achieved

     Number of q in the star =    8
     List of q in the star:
          1  -0.250000000   0.250000000  -0.250000000
          2   0.250000000  -0.250000000  -0.250000000
          3   0.250000000  -0.250000000   0.250000000
          4   0.250000000   0.250000000   0.250000000
          5  -0.250000000  -0.250000000  -0.250000000
          6  -0.250000000  -0.250000000   0.250000000
          7  -0.250000000   0.250000000   0.250000000
          8   0.250000000   0.250000000  -0.250000000

     Diagonalizing the dynamical matrix

     q = (   -0.250000000   0.250000000  -0.250000000 )

 **************************************************************************
     freq (    1) =       3.512771 [THz] =     117.173427 [cm-1]
     freq (    2) =       3.512771 [THz] =     117.173427 [cm-1]
     freq (    3) =       6.338040 [THz] =     211.414258 [cm-1]
 **************************************************************************

     Mode symmetry, C_3v (3m)   point group:

     freq (  1 -  2) =        117.2  [cm-1]   --> E    L_3
     freq (  3 -  3) =        211.4  [cm-1]   --> A_1  L_1
     electron-phonon interaction  ...

     Gaussian Broadening:   0.005 Ry, ngauss=   0
     DOS =  1.339210 states/spin/Ry/Unit Cell at Ef=  8.321793 eV
     lambda(    1)=  0.0023   gamma=    0.04 GHz
     lambda(    2)=  0.0023   gamma=    0.04 GHz
     lambda(    3)=  0.0285   gamma=    1.47 GHz
     Gaussian Broadening:   0.010 Ry, ngauss=   0
     DOS =  1.881761 states/spin/Ry/Unit Cell at Ef=  8.327153 eV
     lambda(    1)=  0.0204   gamma=    0.45 GHz
     lambda(    2)=  0.0207   gamma=    0.46 GHz
     lambda(    3)=  0.2321   gamma=   16.75 GHz
     Gaussian Broadening:   0.015 Ry, ngauss=   0
     DOS =  2.123229 states/spin/Ry/Unit Cell at Ef=  8.328621 eV
     lambda(    1)=  0.0250   gamma=    0.63 GHz
     lambda(    2)=  0.0251   gamma=    0.63 GHz
     lambda(    3)=  0.2280   gamma=   18.57 GHz
     Gaussian Broadening:   0.020 Ry, ngauss=   0
     DOS =  2.249739 states/spin/Ry/Unit Cell at Ef=  8.324319 eV
     lambda(    1)=  0.0283   gamma=    0.75 GHz
     lambda(    2)=  0.0282   gamma=    0.75 GHz
     lambda(    3)=  0.2027   gamma=   17.50 GHz
     Gaussian Broadening:   0.025 Ry, ngauss=   0
     DOS =  2.329803 states/spin/Ry/Unit Cell at Ef=  8.317861 eV
     lambda(    1)=  0.0323   gamma=    0.89 GHz
     lambda(    2)=  0.0322   gamma=    0.88 GHz
     lambda(    3)=  0.1880   gamma=   16.81 GHz
     Gaussian Broadening:   0.030 Ry, ngauss=   0
     DOS =  2.396029 states/spin/Ry/Unit Cell at Ef=  8.311296 eV
     lambda(    1)=  0.0366   gamma=    1.03 GHz
     lambda(    2)=  0.0365   gamma=    1.03 GHz
     lambda(    3)=  0.1841   gamma=   16.92 GHz
     Gaussian Broadening:   0.035 Ry, ngauss=   0
     DOS =  2.455226 states/spin/Ry/Unit Cell at Ef=  8.305262 eV
     lambda(    1)=  0.0408   gamma=    1.18 GHz
     lambda(    2)=  0.0408   gamma=    1.18 GHz
     lambda(    3)=  0.1873   gamma=   17.64 GHz
     Gaussian Broadening:   0.040 Ry, ngauss=   0
     DOS =  2.507873 states/spin/Ry/Unit Cell at Ef=  8.299955 eV
     lambda(    1)=  0.0448   gamma=    1.33 GHz
     lambda(    2)=  0.0449   gamma=    1.33 GHz
     lambda(    3)=  0.1946   gamma=   18.72 GHz
     Gaussian Broadening:   0.045 Ry, ngauss=   0
     DOS =  2.552966 states/spin/Ry/Unit Cell at Ef=  8.295411 eV
     lambda(    1)=  0.0485   gamma=    1.46 GHz
     lambda(    2)=  0.0485   gamma=    1.46 GHz
     lambda(    3)=  0.2039   gamma=   19.97 GHz
     Gaussian Broadening:   0.050 Ry, ngauss=   0
     DOS =  2.589582 states/spin/Ry/Unit Cell at Ef=  8.291553 eV
     lambda(    1)=  0.0517   gamma=    1.58 GHz
     lambda(    2)=  0.0516   gamma=    1.57 GHz
     lambda(    3)=  0.2137   gamma=   21.23 GHz


     Number of q in the star =    8
     List of q in the star:
          1  -0.250000000   0.250000000  -0.250000000
          2   0.250000000  -0.250000000  -0.250000000
          3   0.250000000  -0.250000000   0.250000000
          4   0.250000000   0.250000000   0.250000000
          5  -0.250000000  -0.250000000  -0.250000000
          6  -0.250000000  -0.250000000   0.250000000
          7  -0.250000000   0.250000000   0.250000000
          8   0.250000000   0.250000000  -0.250000000

     Calculation of q =    0.5000000  -0.5000000   0.5000000"""


def test_read_espresso_ph_complex():
    fd = StringIO(complex_ph_output)
    results = read_espresso_ph(fd)

    assert len(results) == 3
    assert results[1]["qpoint"] == (0, 0, 0)
    assert np.unique(results[1]["freqs"]).shape[0] == 1
    assert np.unique(results[1]["freqs"])[0] == 0.173268
    assert len(results[1]["eqpoints"]) == 1
    assert results[1]["atoms"].symbols == ["Al"]

    assert results[2]["qpoint"] == (-0.25, 0.25, -0.25)
    assert np.unique(results[2]["freqs"]).shape[0] == 2
    assert np.unique(results[2]["freqs"])[1] == 6.338040
    assert len(results[2]["eqpoints"]) == 8
    assert results[2]["atoms"].symbols == ["Al"]

    for i in np.arange(0.005, 0.055, 0.005):
        assert results[2]["ep_data"][round(i, 3)]

    assert results[2]["ep_data"][0.005] == {
        "dos": 1.339210,
        "fermi": 8.321793,
        "lambdas": [0.0023, 0.0023, 0.0285],
        "gammas": [0.04, 0.04, 1.47],
    }
