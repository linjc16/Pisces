"""Tests for the Castep.read method"""
from io import StringIO

import numpy as np
import pytest

from ase.constraints import FixAtoms, FixCartesian
from ase.io.castep.castep_reader import (
    _read_forces,
    _read_fractional_coordinates,
    _read_header,
    _read_hirshfeld_charges,
    _read_hirshfeld_details,
    _read_mulliken_charges,
    _read_stress,
    _read_unit_cell,
    _set_energy_and_free_energy,
    read_castep_castep,
)
from ase.units import GPa

HEADER = """\
 ************************************ Title ************************************


 ***************************** General Parameters ******************************

 output verbosity                               : normal  (1)
 write checkpoint data to                       : castep.check
 type of calculation                            : single point energy
 stress calculation                             : off
 density difference calculation                 : off
 electron localisation func (ELF) calculation   : off
 Hirshfeld analysis                             : off
 polarisation (Berry phase) analysis            : off
 molecular orbital projected DOS                : off
 deltaSCF calculation                           : off
 unlimited duration calculation
 timing information                             : on
 memory usage estimate                          : on
 write extra output files                       : on
 write final potential to formatted file        : off
 write final density to formatted file          : off
 write BibTeX reference list                    : on
 write OTFG pseudopotential files               : on
 write electrostatic potential file             : on
 write bands file                               : on
 checkpoint writing                             : both castep_bin and check files
 random number generator seed                   :         42

 *********************** Exchange-Correlation Parameters ***********************

 using functional                               : Local Density Approximation
 DFT+D: Semi-empirical dispersion correction    : off

 ************************* Pseudopotential Parameters **************************

 pseudopotential representation                 : reciprocal space
 <beta|phi> representation                      : reciprocal space
 spin-orbit coupling                            : off

 **************************** Basis Set Parameters *****************************

 basis set accuracy                             : FINE
 finite basis set correction                    : none

 **************************** Electronic Parameters ****************************

 number of  electrons                           :  8.000
 net charge of system                           :  0.000
 treating system as non-spin-polarized
 number of bands                                :          8

 ********************* Electronic Minimization Parameters **********************

 Method: Treating system as metallic with density mixing treatment of electrons,
         and number of  SD  steps               :          1
         and number of  CG  steps               :          4

 total energy / atom convergence tol.           : 0.1000E-04   eV
 eigen-energy convergence tolerance             : 0.1000E-05   eV
 max force / atom convergence tol.              : ignored
 periodic dipole correction                     : NONE

 ************************** Density Mixing Parameters **************************

 density-mixing scheme                          : Broyden
 max. length of mixing history                  :         20

 *********************** Population Analysis Parameters ************************

 Population analysis with cutoff                :  3.000       A
 Population analysis output                     : summary and pdos components

 *******************************************************************************
"""  # noqa: E501

# Some keyword in the .param file triggers a more detailed header.
HEADER_DETAILED = """\
 ************************************ Title ************************************


 ***************************** General Parameters ******************************

 output verbosity                               : normal  (1)
 write checkpoint data to                       : castep.check
 type of calculation                            : single point energy
 stress calculation                             : off
 density difference calculation                 : off
 electron localisation func (ELF) calculation   : off
 Hirshfeld analysis                             : off
 polarisation (Berry phase) analysis            : off
 molecular orbital projected DOS                : off
 deltaSCF calculation                           : off
 unlimited duration calculation
 timing information                             : on
 memory usage estimate                          : on
 write extra output files                       : on
 write final potential to formatted file        : off
 write final density to formatted file          : off
 write BibTeX reference list                    : on
 write OTFG pseudopotential files               : on
 write electrostatic potential file             : on
 write bands file                               : on
 checkpoint writing                             : both castep_bin and check files

 output         length unit                     : A
 output           mass unit                     : amu
 output           time unit                     : ps
 output         charge unit                     : e
 output           spin unit                     : hbar/2
 output         energy unit                     : eV
 output          force unit                     : eV/A
 output       velocity unit                     : A/ps
 output       pressure unit                     : GPa
 output     inv_length unit                     : 1/A
 output      frequency unit                     : cm-1
 output force constant unit                     : eV/A**2
 output         volume unit                     : A**3
 output   IR intensity unit                     : (D/A)**2/amu
 output         dipole unit                     : D
 output         efield unit                     : eV/A/e
 output        entropy unit                     : J/mol/K
 output    efield chi2 unit                     : pm/V

 wavefunctions paging                           : none
 random number generator seed                   :   90945350
 data distribution                              : optimal for this architecture
 optimization strategy                          : balance speed and memory

 *********************** Exchange-Correlation Parameters ***********************

 using functional                               : Local Density Approximation
 relativistic treatment                         : Koelling-Harmon
 DFT+D: Semi-empirical dispersion correction    : off

 ************************* Pseudopotential Parameters **************************

 pseudopotential representation                 : reciprocal space
 <beta|phi> representation                      : reciprocal space
 spin-orbit coupling                            : off

 **************************** Basis Set Parameters *****************************

 plane wave basis set cut-off                   :   180.0000   eV
 size of standard grid                          :     1.7500
 size of   fine   gmax                          :    12.0285   1/A
 finite basis set correction                    : none

 **************************** Electronic Parameters ****************************

 number of  electrons                           :  8.000
 net charge of system                           :  0.000
 treating system as non-spin-polarized
 number of bands                                :          8

 ********************* Electronic Minimization Parameters **********************

 Method: Treating system as metallic with density mixing treatment of electrons,
         and number of  SD  steps               :          1
         and number of  CG  steps               :          4

 total energy / atom convergence tol.           : 0.1000E-04   eV
 eigen-energy convergence tolerance             : 0.1000E-05   eV
 max force / atom convergence tol.              : ignored
 convergence tolerance window                   :          3   cycles
 max. number of SCF cycles                      :         30
 number of fixed-spin iterations                :         10
 smearing scheme                                : Gaussian
 smearing width                                 : 0.2000       eV
 Fermi energy convergence tolerance             : 0.2721E-13   eV
 periodic dipole correction                     : NONE

 ************************** Density Mixing Parameters **************************

 density-mixing scheme                          : Broyden
 max. length of mixing history                  :         20
 charge density mixing amplitude                : 0.8000
 cut-off energy for mixing                      :  180.0       eV

 *********************** Population Analysis Parameters ************************

 Population analysis with cutoff                :  3.000       A
 Population analysis output                     : summary and pdos components

 *******************************************************************************
"""  # noqa: E501


# Some XC functionals cannot be mapped by ASE to keywords; e.g. from Castep 25
HEADER_PZ_LDA = """\
 ************************************ Title ************************************
 

 ***************************** General Parameters ******************************
  
 output verbosity                               : normal  (1)
 write checkpoint data to                       : castep.check
 type of calculation                            : single point energy
 stress calculation                             : on
 density difference calculation                 : off
 electron localisation func (ELF) calculation   : off
 Hirshfeld analysis                             : off
 polarisation (Berry phase) analysis            : off
 molecular orbital projected DOS                : off
 deltaSCF calculation                           : off
 unlimited duration calculation
 timing information                             : on
 memory usage estimate                          : on
 write extra output files                       : on
 write final potential to formatted file        : off
 write final density to formatted file          : off
 write BibTeX reference list                    : on
 write OTFG pseudopotential files               : on
 write electrostatic potential file             : on
 write bands file                               : on
 checkpoint writing                             : both castep_bin and check files
 random number generator seed                   :  112211524

 *********************** Exchange-Correlation Parameters ***********************
  
 using functional                               : Perdew-Zunger Local Density Approximation
 DFT+D: Semi-empirical dispersion correction    : off

 ************************* Pseudopotential Parameters **************************
  
 pseudopotential representation                 : reciprocal space
 <beta|phi> representation                      : reciprocal space
 spin-orbit coupling                            : off

 **************************** Basis Set Parameters *****************************
  
 basis set accuracy                             : FINE
 finite basis set correction                    : none

 **************************** Electronic Parameters ****************************
  
 number of  electrons                           :  28.00    
 net charge of system                           :  0.000    
 treating system as non-spin-polarized
 number of bands                                :         18

 ********************* Electronic Minimization Parameters **********************
  
 Method: Treating system as metallic with density mixing treatment of electrons,
         and number of  SD  steps               :          1
         and number of  CG  steps               :          4
  
 total energy / atom convergence tol.           : 0.1000E-04   eV
 eigen-energy convergence tolerance             : 0.1429E-06   eV
 max force / atom convergence tol.              : ignored
 periodic dipole correction                     : NONE

 ************************** Density Mixing Parameters **************************
  
 density-mixing scheme                          : Pulay
 max. length of mixing history                  :         20

 *********************** Population Analysis Parameters ************************
  
 Population analysis with cutoff                :  3.000       A
 Population analysis output                     : summary and pdos components

 *******************************************************************************

"""  # noqa: E501,W291,W293


def test_header():
    """Test if the header blocks can be parsed correctly."""
    out = StringIO(HEADER)
    parameters = _read_header(out)
    parameters_ref = {
        'task': 'SinglePoint',
        'iprint': 1,
        'calculate_stress': False,
        'xc_functional': 'LDA',
        'sedc_apply': False,
        'basis_precision': 'FINE',
        'finite_basis_corr': 0,
        'elec_energy_tol': 1e-5,
        'mixing_scheme': 'Broyden',
    }
    assert parameters == parameters_ref


def test_header_detailed():
    """Test if the header blocks can be parsed correctly."""
    out = StringIO(HEADER_DETAILED)
    parameters = _read_header(out)
    parameters_ref = {
        'task': 'SinglePoint',
        'iprint': 1,
        'calculate_stress': False,
        'opt_strategy': 'Default',
        'xc_functional': 'LDA',
        'sedc_apply': False,
        'cut_off_energy': 180.0,
        'finite_basis_corr': 0,
        'elec_energy_tol': 1e-5,
        'elec_convergence_win': 3,
        'mixing_scheme': 'Broyden',
    }
    assert parameters == parameters_ref


def test_header_castep25():
    """Test if header block with unknown XC functional is parsed correctly"""
    out = StringIO(HEADER_PZ_LDA)
    parameters = _read_header(out)
    parameters_ref = {
        "task": "SinglePoint",
        "iprint": 1,
        "calculate_stress": True,
        "xc_functional": "Perdew-Zunger Local Density Approximation",
        "sedc_apply": False,
        "basis_precision": "FINE",
        "finite_basis_corr": 0,
        "elec_energy_tol": 1e-5,
        "mixing_scheme": "Pulay",
    }
    assert parameters == parameters_ref


UNIT_CELL = """\
                           -------------------------------
                                      Unit Cell
                           -------------------------------
        Real Lattice(A)              Reciprocal Lattice(1/A)
    -0.0287130     2.6890780     2.6889378       -1.148709953   1.163162213   1.161190328
     2.6890780    -0.0287130     2.6889378        1.163162214  -1.148709951   1.161190326
     2.6938401     2.6938401    -0.0335277        1.159077172   1.159077170  -1.146760802
"""  # noqa: E501


def test_unit_cell():
    """Test if the Unit Cell block can be parsed correctly."""
    out = StringIO(UNIT_CELL)
    out.readline()
    out.readline()
    cell = _read_unit_cell(out)
    cell_ref = [
        [-0.0287130, +2.6890780, +2.6889378],
        [+2.6890780, -0.0287130, +2.6889378],
        [+2.6938401, +2.6938401, -0.0335277],
    ]
    np.testing.assert_allclose(cell, cell_ref)


FRACTIONAL_COORDINATES = """\
            xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            x  Element         Atom        Fractional coordinates of atoms  x
            x                 Number           u          v          w      x
            x---------------------------------------------------------------x
            x  Si                1        -0.000000  -0.000000   0.000000   x
            x  Si                2         0.249983   0.249983   0.254121   x
            xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""


def test_fractional_coordinates():
    """Test if fractional coordinates can be parsed correctly."""
    out = StringIO(FRACTIONAL_COORDINATES)
    out.readline()
    out.readline()
    species, custom_species, positions_frac = \
        _read_fractional_coordinates(out, 2)
    positions_frac_ref = [
        [-0.000000, -0.000000, -0.000000],
        [+0.249983, +0.249983, +0.254121],
    ]
    np.testing.assert_array_equal(species, ['Si', 'Si'])
    assert custom_species is None
    np.testing.assert_allclose(positions_frac, positions_frac_ref)


FORCES = """\
 ************************** Forces **************************
 *                                                          *
 *               Cartesian components (eV/A)                *
 * -------------------------------------------------------- *
 *                         x            y            z      *
 *                                                          *
 * Si              1     -0.02211     -0.02210     -0.02210 *
 * Si              2      0.02211      0.02210      0.02210 *
 *                                                          *
 ************************************************************
 """

CONSTRAINED_FORCES = """\
 ******************************** Constrained Forces ********************************
 *                                                                                  *
 *                           Cartesian components (eV/A)                            *
 * -------------------------------------------------------------------------------- *
 *                         x                    y                    z              *
 *                                                                                  *
 * Si              1      0.00000(cons'd)      0.00000(cons'd)      0.00000(cons'd) *
 * Si              2     -0.00252             -0.00252              0.00000(cons'd) *
 *                                                                                  *
 ************************************************************************************
"""  # noqa: E501


def test_forces():
    """Test if the Forces block can be parsed correctly."""
    out = StringIO(FORCES)
    out.readline()
    forces, constraints = _read_forces(out, n_atoms=2)
    forces_ref = [
        [-0.02211, -0.02210, -0.02210],
        [+0.02211, +0.02210, +0.02210],
    ]
    np.testing.assert_allclose(forces, forces_ref)
    assert not constraints


def test_constrainted_forces():
    """Test if the Constrainted Forces block can be parsed correctly."""
    out = StringIO(CONSTRAINED_FORCES)
    out.readline()
    forces, constraints = _read_forces(out, n_atoms=2)
    forces_ref = [
        [+0.00000, +0.00000, +0.00000],
        [-0.00252, -0.00252, +0.00000],
    ]
    constraints_ref = [
        FixAtoms(0),
        FixCartesian(1, mask=(0, 0, 1)),
    ]
    np.testing.assert_allclose(forces, forces_ref)
    assert all(constraints[0].index == constraints_ref[0].index)
    assert all(constraints[1].index == constraints_ref[1].index)
    assert all(constraints[1].mask == constraints_ref[1].mask)


STRESS = """\
 ***************** Stress Tensor *****************
 *                                               *
 *          Cartesian components (GPa)           *
 * --------------------------------------------- *
 *             x             y             z     *
 *                                               *
 *  x     -0.006786     -0.035244      0.023931  *
 *  y     -0.035244     -0.006786      0.023931  *
 *  z      0.023931      0.023931     -0.011935  *
 *                                               *
 *  Pressure:    0.0085                          *
 *                                               *
 *************************************************
"""


def test_stress():
    """Test if the Stress Tensor block can be parsed correctly."""
    out = StringIO(STRESS)
    out.readline()
    results = _read_stress(out)
    stress_ref = [
        -0.006786,
        -0.006786,
        -0.011935,
        +0.023931,
        +0.023931,
        -0.035244,
    ]
    stress_ref = np.array(stress_ref) * GPa
    pressure_ref = 0.0085 * GPa
    np.testing.assert_allclose(results['stress'], stress_ref)
    np.testing.assert_allclose(results['pressure'], pressure_ref)


# bulk("AlP", "zincblende", a=5.43)
MULLIKEN_SPIN_UNPOLARIZED = """\
     Atomic Populations (Mulliken)
     -----------------------------
Species          Ion     s       p       d       f      Total   Charge (e)
==========================================================================
  Al              1     0.935   1.361   0.000   0.000   2.296     0.704
  P               1     1.665   4.039   0.000   0.000   5.704    -0.704
==========================================================================
"""

# bulk("MnTe", "zincblende", a=6.34)
MULLIKEN_SPIN_POLARIZED = """\
     Atomic Populations (Mulliken)
     -----------------------------
Species          Ion Spin      s       p       d       f      Total   Charge(e)   Spin(hbar/2)
==============================================================================================
  Mn              1   up:     1.436   3.596   4.918   0.000   9.950    -0.114        4.785
                  1   dn:     1.293   3.333   0.538   0.000   5.164
  Te              1   up:     0.701   2.229   0.000   0.000   2.929     0.114       -0.027
                  1   dn:     0.763   2.194   0.000   0.000   2.956
==============================================================================================
"""  # noqa: E501


def test_mulliken_spin_unpolarized():
    """Test if the Atomic Populations block can be parsed correctly."""
    out = StringIO(MULLIKEN_SPIN_UNPOLARIZED)
    out.readline()  # read header
    results = _read_mulliken_charges(out)
    np.testing.assert_allclose(results['charges'], [+0.704, -0.704])
    assert 'magmoms' not in results


def test_mulliken_spin_polarized():
    """Test if the Atomic Populations block can be parsed correctly."""
    out = StringIO(MULLIKEN_SPIN_POLARIZED)
    out.readline()  # read header
    results = _read_mulliken_charges(out)
    np.testing.assert_allclose(results['charges'], [-0.114, +0.114])
    np.testing.assert_allclose(results['magmoms'], [+4.785, -0.027])


HIRSHFELD_DETAILS = """\
 Species     1,  Atom     1  :  Al
  Fractional coordinates :
                                        0.000000000   0.000000000   0.000000000
  Cartesian coordinates (A) :
                                        0.000000000   0.000000000   0.000000000
  Free atom total nuclear charge (e) :
                                        3.000000000
  Free atom total electronic charge on real space grid (e) :
                                       -3.000000000
  SCF total electronic charge on real space grid (e) :
                                       -8.000000000
  cut-off radius for r-integrals :
                                       10.000000000
  Free atom volume (Bohr**3) :
                                       67.048035000
  Hirshfeld total electronic charge (e) :
                                       -2.821040742
  Hirshfeld net atomic charge (e) :
                                        0.178959258
  Hirshfeld atomic volume (Bohr**3) :
                                       61.353953500
  Hirshfeld / free atomic volume :
                                        0.915074595

 Species     2,  Atom     1  :  P
  Fractional coordinates :
                                        0.250000000   0.250000000   0.250000000
  Cartesian coordinates (A) :
                                        1.357500000   1.357500000   1.357500000
  Free atom total nuclear charge (e) :
                                        5.000000000
  Free atom total electronic charge on real space grid (e) :
                                       -5.000000000
  SCF total electronic charge on real space grid (e) :
                                       -8.000000000
  cut-off radius for r-integrals :
                                       10.000000000
  Free atom volume (Bohr**3) :
                                       70.150468179
  Hirshfeld total electronic charge (e) :
                                       -5.178959258
  Hirshfeld net atomic charge (e) :
                                       -0.178959258
  Hirshfeld atomic volume (Bohr**3) :
                                       66.452900385
  Hirshfeld / free atomic volume :
                                        0.947290904

"""


def test_hirshfeld_details():
    """Test if the Hirshfeld block of ispin > 1 can be parsed correctly."""
    out = StringIO(HIRSHFELD_DETAILS)
    results = _read_hirshfeld_details(out, 2)
    np.testing.assert_allclose(
        results['hirshfeld_volume_ratios'],
        [0.915074595, 0.947290904],
    )


HIRSHFELD_SPIN_UNPOLARIZED = """\
     Hirshfeld Analysis
     ------------------
Species   Ion     Hirshfeld Charge (e)
======================================
  Al       1                 0.18
  P        1                -0.18
======================================
"""

HIRSHFELD_SPIN_POLARIZED = """\
     Hirshfeld Analysis
     ------------------
Species   Ion     Hirshfeld Charge (e)  Spin (hbar/2)
===================================================
  Mn       1                 0.06        4.40
  Te       1                -0.06        0.36
===================================================
"""


def test_hirshfeld_spin_unpolarized():
    """Test if the Hirshfeld Analysis block can be parsed correctly."""
    out = StringIO(HIRSHFELD_SPIN_UNPOLARIZED)
    out.readline()  # read header
    results = _read_hirshfeld_charges(out)
    np.testing.assert_allclose(results['hirshfeld_charges'], [+0.18, -0.18])
    assert 'hirshfeld_magmoms' not in results


def test_hirshfeld_spin_polarized():
    """Test if the Hirshfeld Analysis block can be parsed correctly."""
    out = StringIO(HIRSHFELD_SPIN_POLARIZED)
    out.readline()  # read header
    results = _read_hirshfeld_charges(out)
    np.testing.assert_allclose(results['hirshfeld_charges'], [+0.06, -0.06])
    np.testing.assert_allclose(results['hirshfeld_magmoms'], [+4.40, +0.36])


def test_energy_and_free_energy_metallic():
    """Test if `energy` and `free_energy` is set correctly.

    This test is made based on the following output obtained for Si.
    ```
    Final energy, E             =  -340.9490879813     eV
    Final free energy (E-TS)    =  -340.9490902954     eV
    (energies not corrected for finite basis set)

    NB est. 0K energy (E-0.5TS)      =  -340.9490891384     eV

    (SEDC) Total Energy Correction : -0.567289E+00 eV

    Dispersion corrected final energy*, Ecor          =  -341.5163768370     eV
    Dispersion corrected final free energy* (Ecor-TS) =  -341.5163791511     eV
    NB dispersion corrected est. 0K energy* (Ecor-0.5TS) =  -341.5163779940     eV
     For future reference: finite basis dEtot/dlog(Ecut) =      -0.487382eV
     Total energy corrected for finite basis set =    -341.516014 eV
    ```
    """  # noqa: E501
    results = {
        'energy_without_dispersion_correction': -340.9490879813,
        'free_energy_without_dispersion_correction': -340.9490902954,
        'energy_zero_without_dispersion_correction': -340.9490891384,
    }
    results.update(results)
    _set_energy_and_free_energy(results)
    assert results['energy'] == -340.9490891384
    assert results['free_energy'] == -340.9490902954

    results_dispersion_correction = {
        'energy_with_dispersion_correction': -341.5163768370,
        'free_energy_with_dispersion_correction': -341.5163791511,
        'energy_zero_with_dispersion_correction': -341.5163779940,
    }
    results.update(results_dispersion_correction)
    _set_energy_and_free_energy(results)
    assert results['energy'] == -341.5163779940
    assert results['free_energy'] == -341.5163791511

    results.update({'energy_with_finite_basis_set_correction': -341.516014})
    _set_energy_and_free_energy(results)
    assert results['energy'] == -341.5163779940
    assert results['free_energy'] == -341.5163791511


def test_energy_and_free_energy_non_metallic():
    """Test if `energy` and `free_energy` is set correctly.

    This test is made based on the following output obtained for Si.
    ```
    Final energy =  -340.9491006696     eV
    (energy not corrected for finite basis set)

    (SEDC) Total Energy Correction : -0.567288E+00 eV

    Dispersion corrected final energy* =  -341.5163888035     eV
     For future reference: finite basis dEtot/dlog(Ecut) =      -0.487570eV
     Total energy corrected for finite basis set =    -341.516024 eV
    ```
    """
    results = {'energy_without_dispersion_correction': -340.9491006696}
    _set_energy_and_free_energy(results)
    assert results['energy'] == -340.9491006696
    assert results['free_energy'] == -340.9491006696

    results.update({'energy_with_dispersion_correction': -341.5163888035})
    _set_energy_and_free_energy(results)
    assert results['energy'] == -341.5163888035
    assert results['free_energy'] == -341.5163888035

    results.update({'energy_with_finite_basis_set_correction': -341.516024})
    _set_energy_and_free_energy(results)
    assert results['energy'] == -341.516024
    assert results['free_energy'] == -341.5163888035


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_md_images(datadir):
    """Test if multiple images can be read for the MolecularDynamics task."""
    images = read_castep_castep(f'{datadir}/castep/md.castep', index=':')

    assert len(images) == 3  # 0th, 1st, 2nd steps

    # `read_castep_castep_old` could parse multi-images but not forces / stress
    # check if new `read_castep_castep` can do

    atoms = images[-1]

    forces_ref = [
        [-0.09963, +0.10729, -0.00665],
        [+0.09339, +0.01482, +0.01147],
        [-0.02828, -0.11817, -0.03557],
        [+0.14991, -0.01604, +0.02132],
        [-0.03495, +0.04016, -0.03526],
        [+0.07136, -0.07883, +0.04578],
        [-0.26870, -0.12445, -0.14684],
        [+0.11690, +0.17523, +0.14574],
    ]
    np.testing.assert_array_almost_equal(atoms.get_forces(), forces_ref)

    stress_ref = [
        +0.390672, +0.388091, +0.385978, -0.276337, -0.061901, -0.144025,
    ]
    np.testing.assert_array_almost_equal(atoms.get_stress() / GPa, stress_ref)
