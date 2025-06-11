"""Test file for exciting ASE calculator."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

import ase
import ase.calculators.exciting.exciting
import ase.calculators.exciting.runner

# Note this is an imitation of an exciting INFO.out output file.
# We've removed many of the lines of text that were originally in this outfile
# that are note usefule for testing purposes to save space in this file.
# We've also modified the file to contain a Ti atom and use an HCP cell to
# make the test more interesting since the HCP cell gives non-symmetric cell
# vectors in a cartesian basis set.

LDA_VWN_AR_INFO_OUT = """
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ Starting initialization                                                      +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

 Lattice vectors (cartesian) :
     10.3360193975     10.3426010725      0.0054547264
    -10.3461511392     10.3527307290      0.0059928210
     10.3354645037     10.3540072605     20.6246241525

 Reciprocal lattice vectors (cartesian) :
      0.3039381122      0.3039214341     -0.3048853768
     -0.3036485697      0.3034554311     -0.0001760382
      0.0000078456     -0.0001685540      0.3047255253

 Unit cell volume                           :    4412.7512103067
 Brillouin zone volume                      :       0.0562121456

 Species :    1 (Ti)
     parameters loaded from                 :    Ti.xml
     name                                   :    titanium

     atomic positions (lattice) :
       1 :   0.00000000  0.00000000  0.00000000

 Total number of atoms per unit cell        :       1

 Spin treatment                             :    spin-unpolarised

 Number of Bravais lattice symmetries       :      48
 Number of crystal symmetries               :      48

 k-point grid                               :       1    1    1
 Total number of k-points                   :       1
 k-point set is reduced with crystal symmetries

 R^MT_min * |G+k|_max (rgkmax)              :      10.00000000
 Species with R^MT_min                      :       1 (Ti)
 Maximum |G+k| for APW functions            :       1.66666667
 Maximum |G| for potential and density      :       7.50000000
 Polynomial order for pseudochg. density    :       9

 G-vector grid sizes                        :      36    36    36
 Total number of G-vectors                  :   23871

 Maximum angular momentum used for
     APW functions                          :       8
     computing H and O matrix elements      :       4
     potential and density                  :       4
     inner part of muffin-tin               :       2

 Total nuclear charge                       :     -22.00000000
 Total electronic charge                    :      22.00000000
 Total core charge                          :      18.00000000
 Total valence charge                       :       4.00000000

 Effective Wigner radius, r_s               :       3.55062021

 Number of empty states                     :       5
 Total number of valence states             :      10

 Maximum Hamiltonian size                   :     263
 Maximum number of plane-waves              :     251
 Total number of local-orbitals             :      12

 Exchange-correlation type                  :     100
     libxc; exchange: Slater exchange; correlation: Vosko, Wilk & Nusair (VWN5)

 Smearing scheme                            :    Gaussian
 Smearing width                             :       0.00100000

 Using multisecant Broyden potential mixing

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ Ending initialization                                                        +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ SCF iteration number :    1                                                  +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Total energy                               :      -527.82493279
 _______________________________________________________________
 Fermi energy                               :        -0.20111449
 Kinetic energy                             :       530.56137212
 Coulomb energy                             :     -1029.02167746
 Exchange energy                            :       -27.93377198
 Correlation energy                         :        -1.43085548
 Sum of eigenvalues                         :      -305.07886015
 Effective potential energy                 :      -835.64023227
 Coulomb potential energy                   :      -796.81322609
 xc potential energy                        :       -38.82700618
 Hartree energy                             :       205.65681157
 Electron-nuclear energy                    :     -1208.12684923
 Nuclear-nuclear energy                     :       -26.55163980
 Madelung energy                            :      -630.61506441
 Core-electron kinetic energy               :         0.00000000

 DOS at Fermi energy (states/Ha/cell)       :         0.00000000

 Electron charges :
     core                                   :        10.00000000
     core leakage                           :         0.00000000
     valence                                :         8.00000000
     interstitial                           :         0.00183897
     charge in muffin-tin spheres :
                  atom     1    Ar          :        17.99816103
     total charge in muffin-tins            :        17.99816103
     total charge                           :        18.00000000

 Estimated fundamental gap                  :         0.36071248
        valence-band maximum at    1      0.0000  0.0000  0.0000
     conduction-band minimum at    1      0.0000  0.0000  0.0000

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
| Convergency criteria checked for the last 2 iterations                       +
| Convergence targets achieved. Performing final SCF iteration                 +
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Total energy                               :      -527.81796101
 _______________________________________________________________
 Fermi energy                               :        -0.20044598
 Kinetic energy                             :       530.57303096
 Coulomb energy                             :     -1029.02642037
 Exchange energy                            :       -27.93372809
 Correlation energy                         :        -1.43084350
 Sum of eigenvalues                         :      -305.07413840
 Effective potential energy                 :      -835.64716936
 Coulomb potential energy                   :      -796.82023455
 xc potential energy                        :       -38.82693481
 Hartree energy                             :       205.65454603
 Electron-nuclear energy                    :     -1208.12932661
 Nuclear-nuclear energy                     :       -26.55163980
 Madelung energy                            :      -630.61630310
 Core-electron kinetic energy               :         0.00000000

 DOS at Fermi energy (states/Ha/cell)       :         0.00000000

 Electron charges :
     core                                   :        10.00000000
     core leakage                           :         0.00000000
     valence                                :         8.00000000
     interstitial                           :         0.00184037
     charge in muffin-tin spheres :
                  atom     1    Ar          :        17.99815963
     total charge in muffin-tins            :        17.99815963
     total charge                           :        18.00000000

 Estimated fundamental gap                  :         0.36095838
        valence-band maximum at    1      0.0000  0.0000  0.0000
     conduction-band minimum at    1      0.0000  0.0000  0.0000

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+ Self-consistent loop stopped                                                 +
| EXCITING NITROGEN-14 stopped                                                 =
"""


@pytest.fixture()
def nitrogen_trioxide_atoms():
    """Pytest fixture that creates ASE Atoms cell for other tests."""
    return ase.Atoms('NO3',
                     cell=[[2, 2, 0], [0, 4, 0], [0, 0, 6]],
                     scaled_positions=[(0, 0, 0), (0.25, 0.25, 0),
                                       (0, 0, 0.75), (0.5, 0.5, 0.5)],
                     pbc=True)


def test_ground_state_template_init(excitingtools):
    """Test initialization of the ExcitingGroundStateTemplate class."""
    gs_template_obj = (
        ase.calculators.exciting.exciting.ExcitingGroundStateTemplate())
    assert gs_template_obj.name == 'exciting'
    assert len(gs_template_obj.implemented_properties) == 2
    assert 'energy' in gs_template_obj.implemented_properties


def test_ground_state_template_write_input(
        tmp_path, nitrogen_trioxide_atoms, excitingtools):
    """Test the write input method of ExcitingGroundStateTemplate.

    We test is by writing a ground state calculation and a bandstructure
    calculation after that is run.

    Args:
        tmp_path: This tells pytest to create a temporary directory
             in which we will store the exciting input file.
        nitrogen_trioxide_atoms: pytest fixture to create ASE Atoms
            unit cell composed of NO3.
    """
    from excitingtools.input.bandstructure import (
        band_structure_input_from_ase_atoms_obj,
    )
    expected_path = tmp_path / 'input.xml'
    # Expected number of points in the bandstructure.
    expected_number_of_special_points = 12
    bandstructure_steps = 100
    binary_path = tmp_path / 'exciting_binary'

    gs_template_obj = (
        ase.calculators.exciting.exciting.ExcitingGroundStateTemplate())
    exciting_profile = ase.calculators.exciting.exciting.ExcitingProfile(
        command=str(binary_path))
    gs_template_obj.write_input(
        profile=exciting_profile,
        directory=tmp_path,
        atoms=nitrogen_trioxide_atoms,
        parameters={
            'title': None,
            'species_path': tmp_path,
            'ground_state_input': {
                'rgkmax': 8.0,
                'do': 'fromscratch',
                "ngridk": [6, 6, 6],
                'xctype': 'GGA_PBE_SOL',
                'vkloff': [0, 0, 0]},
            'properties_input': {
                'bandstructure': band_structure_input_from_ase_atoms_obj(
                    nitrogen_trioxide_atoms, steps=bandstructure_steps)}})
    # Let's assert the file we just wrote exists.
    assert expected_path.exists()
    # Let's assert it's what we expect.
    element_tree = ET.parse(expected_path)
    # Ensure the coordinates of the atoms in the unit cell is correct.
    # We could test the other parts of the input file related coming from
    # the ASE Atoms object like species data but this is tested already in
    # test/io/exciting/test_exciting.py.
    coords_list = element_tree.findall('./structure/species/atom')
    positions = np.array([[float(x)
                           for x in coords_list[i].get('coord').split()]
                          for i in range(len(coords_list))])
    assert positions == pytest.approx(
        nitrogen_trioxide_atoms.get_scaled_positions())

    # Ensure that the exciting calculator properites (e.g. functional type have
    # been set).
    assert element_tree.findall('input') is not None
    assert element_tree.getroot().tag == 'input'
    assert element_tree.getroot()[2].attrib['xctype'] == 'GGA_PBE_SOL'
    assert element_tree.getroot()[2].attrib['rgkmax'] == '8.0'
    # Ensure the bandstructure path is correct:
    band_path = element_tree.findall(
        './properties/bandstructure/plot1d/path')[0]
    assert band_path.tag == 'path'
    assert int(band_path.get('steps')) == bandstructure_steps
    assert len(list(band_path)) == expected_number_of_special_points


def test_ground_state_template_read_results(tmp_path, excitingtools):
    """Test the read result method of ExcitingGroundStateTemplate."""
    # ASE doesn't want us to store any other files for test, so instead
    # we copy an example exciting INFO.out file into the global variable
    # LDA_VWN_AR_INFO_OUT.
    output_file_path = tmp_path / 'info.xml'
    with open(output_file_path, "w", encoding="utf8") as xml_file:
        xml_file.write(LDA_VWN_AR_INFO_OUT)

    gs_template_obj = (
        ase.calculators.exciting.exciting.ExcitingGroundStateTemplate())
    results = gs_template_obj.read_results(tmp_path)
    final_scl_iteration = list(results["scl"].keys())[-1]
    assert pytest.approx(float(results["scl"][
        final_scl_iteration]["Hartree energy"])) == 205.65454603


def test_get_total_energy_and_bandgap(excitingtools):
    """Test getter methods for energy/bandgap results."""
    # Create a fake results dictionary that has two SCL cycles
    # and only contains values for the total energy and bandgap.
    results_dict = {
        'scl': {
            '1':
                {
                    'Total energy': '-240.3',
                    'Estimated fundamental gap': 2.0,
                },
            '2':
                {
                    'Total energy': '-242.3',
                    'Estimated fundamental gap': 3.1,
                }
        }

    }
    results_obj = ase.calculators.exciting.exciting.ExcitingGroundStateResults(
        results_dict)
    assert pytest.approx(results_obj.total_energy()) == -242.3
    assert pytest.approx(results_obj.band_gap()) == 3.1


def test_ground_state_calculator_init(tmpdir, excitingtools):
    """Test initiliazation of the ExcitingGroundStateCalculator"""
    ground_state_input_dict = {
        "rgkmax": 8.0,
        "do": "fromscratch",
        "ngridk": [6, 6, 6],
        "xctype": "GGA_PBE_SOL",
        "vkloff": [0, 0, 0]}
    calc_obj = ase.calculators.exciting.exciting.ExcitingGroundStateCalculator(
        runner=ase.calculators.exciting.runner.SimpleBinaryRunner(
            "exciting_serial", ['./'], 1, tmpdir, ['']),
        ground_state_input=ground_state_input_dict, directory=tmpdir)
    assert calc_obj.parameters["ground_state_input"]["rgkmax"] == 8.0
