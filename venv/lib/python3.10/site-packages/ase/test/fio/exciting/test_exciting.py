"""Test file for exciting file input and output methods."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import ase
import ase.io.exciting

# Import a realistic looking exciting text output file as a string.
from ase.test.calculator.exciting.test_exciting import LDA_VWN_AR_INFO_OUT


@pytest.fixture()
def excitingtools():
    """If we cannot import excitingtools we skip tests with this fixture."""
    return pytest.importorskip('excitingtools')


@pytest.fixture()
def nitrogen_trioxide_atoms():
    """Helper fixture to create a NO3 ase atoms object for tests."""
    return ase.Atoms('NO3',
                     cell=[[2, 2, 0], [0, 4, 0], [0, 0, 6]],
                     positions=[(0, 0, 0), (1, 3, 0),
                                (0, 0, 1), (0.5, 0.5, 0.5)],
                     pbc=True)


def test_write_input_xml_file(
        tmp_path, nitrogen_trioxide_atoms, excitingtools):
    """Test writing input.xml file using write_input_xml_file()."""
    file_path = tmp_path / 'input.xml'
    ground_state_input_dict = {
        "rgkmax": 8.0,
        "do": "fromscratch",
        "ngridk": [6, 6, 6],
        "xctype": "GGA_PBE_SOL",
        "vkloff": [0, 0, 0],
        "tforce": True,
        "nosource": False
    }
    ase.io.exciting.write_input_xml_file(
        file_name=file_path,
        atoms=nitrogen_trioxide_atoms,
        ground_state_input=ground_state_input_dict,
        species_path="/dummy/arbitrary/path",
        title=None)
    assert file_path.exists()
    # Now read the XML file and ensure that it has what we expect:
    atoms_obj = ase.io.exciting.ase_atoms_from_exciting_input_xml(file_path)

    assert all(atoms_obj.symbols == "NOOO")
    input_xml_tree = ET.parse(file_path).getroot()
    parsed_calc_params = list(input_xml_tree)[2]
    assert parsed_calc_params.get("xctype") == "GGA_PBE_SOL"
    assert parsed_calc_params.get("rgkmax") == '8.0'
    assert parsed_calc_params.get("tforce") == 'true'


def test_write_bs_xml(
        tmp_path, nitrogen_trioxide_atoms, excitingtools):
    """Test writing input for bandstructure and skip ground state.

    The input.xml should contain a `do`=skip key in the groun state to avoid
    repeating the ground state. `do`=fromscratch can also be called if the
    ground state should be recalculated or was never done.

    The bandstructure is passed into the exciting xml as an additional property.
    We use excitingtools and pass the ase atoms object and the number of steps
    we want to run to get the bandstructure. Excitingtools in turn calls
    bandpath = cell.bandpath() on the ase atoms object cell (lattice vectors).
    This is done so that excitingtools is independent of ASE.

    """
    from excitingtools.input.bandstructure import (
        band_structure_input_from_ase_atoms_obj,
    )
    file_path = tmp_path / 'input.xml'
    ground_state_input_dict = {
        "rgkmax": 8.0,
        "do": "skip",
        "ngridk": [6, 6, 6],
        "xctype": "GGA_PBE_SOL",
        "vkloff": [0, 0, 0],
        "tforce": True,
        "nosource": False
    }
    bandstructure_steps = 100
    bandstructure = band_structure_input_from_ase_atoms_obj(
        nitrogen_trioxide_atoms, steps=bandstructure_steps)
    properties_input_dict = {'bandstructure': bandstructure}

    ase.io.exciting.write_input_xml_file(
        file_name=file_path,
        atoms=nitrogen_trioxide_atoms,
        ground_state_input=ground_state_input_dict,
        species_path="/dummy/arbitrary/path",
        title=None,
        properties_input=properties_input_dict)
    assert file_path.exists()
    # Now read the XML file and ensure that it has what we expect:
    atoms_obj = ase.io.exciting.ase_atoms_from_exciting_input_xml(file_path)

    assert all(atoms_obj.symbols == "NOOO")
    input_xml_tree = ET.parse(file_path).getroot()
    # Check ground state parameters.
    parsed_ground_state_calc_params = list(input_xml_tree)[2]
    assert parsed_ground_state_calc_params.get("xctype") == "GGA_PBE_SOL"
    assert parsed_ground_state_calc_params.get("rgkmax") == "8.0"
    assert parsed_ground_state_calc_params.get("tforce") == "true"
    assert parsed_ground_state_calc_params.get("do") == "skip"

    # Check additional properties which all relate to the bandstructure.
    parsed_properties = list(input_xml_tree)[3]
    assert parsed_properties.tag == "properties"
    assert len(list(parsed_properties)) == 1
    assert parsed_properties[0].tag == "bandstructure"
    assert parsed_properties[0][0].tag == "plot1d"
    parsed_bandstructure_path = parsed_properties[0][0][0]
    assert parsed_bandstructure_path.tag == "path"
    parsed_bandstructure_path.get("steps") == 100
    parsed_bandstructure_gamma_point = parsed_bandstructure_path[0]
    assert parsed_bandstructure_gamma_point.tag == "point"
    assert parsed_bandstructure_gamma_point.get("coord") == "0.0 0.0 0.0"


def test_write_dos_xml(
        tmp_path, nitrogen_trioxide_atoms, excitingtools):
    """Test creating required input to run a DOS calculation."""
    file_path = tmp_path / 'input.xml'
    ground_state_input_dict = {
        "rgkmax": 8.0,
        "do": "skip",
        "ngridk": [6, 6, 6],
        "xctype": "GGA_PBE_SOL",
        "vkloff": [0, 0, 0],
        "tforce": True,
        "nosource": False
    }
    nsmdos = 2
    ngrdos = 300
    nwdos = 1000
    winddos = [-0.3, 0.3]
    properties_input_dict = {'dos': {
        'nsmdos': nsmdos, 'ngrdos': ngrdos,
        'nwdos': nwdos, 'winddos': winddos}}
    ase.io.exciting.write_input_xml_file(
        file_name=file_path,
        atoms=nitrogen_trioxide_atoms,
        ground_state_input=ground_state_input_dict,
        species_path="/dummy/arbitrary/path",
        title=None,
        properties_input=properties_input_dict)
    assert file_path.exists()
    # Now read the XML file and ensure that it has what we expect:
    atoms_obj = ase.io.exciting.ase_atoms_from_exciting_input_xml(file_path)
    assert all(atoms_obj.symbols == "NOOO")
    input_xml_tree = ET.parse(file_path).getroot()
    # Check ground state parameters.
    parsed_ground_state_calc_params = list(input_xml_tree)[2]
    assert parsed_ground_state_calc_params.get("do") == "skip"
    # Check additional properties which all relate to the bandstructure.
    parsed_properties = list(input_xml_tree)[3]
    assert parsed_properties.tag == "properties"
    assert len(list(parsed_properties)) == 1
    assert len(list(parsed_properties)) == 1
    assert parsed_properties[0].tag == "dos"
    assert parsed_properties[0].get('nsmdos') == str(nsmdos)
    assert parsed_properties[0].get('ngrdos') == str(ngrdos)


def test_ase_atoms_from_exciting_input_xml(
        tmp_path, nitrogen_trioxide_atoms, excitingtools):
    """Test reading the of the exciting input.xml file into ASE atoms obj."""
    expected_cell = np.array([[2, 2, 0], [0, 4, 0], [0, 0, 6]])
    expected_positions = np.array([(0, 0, 0), (1, 3, 0), (0, 0, 1),
                                   (0.5, 0.5, 0.5)])
    # First we write an input.xml file into a temp dir, so we can
    # read it back with our method we put under test.
    file_path = tmp_path / 'input.xml'
    ground_state_input_dict = {
        "rgkmax": 8.0,
        "do": "fromscratch",
        "ngridk": [6, 6, 6],
        "xctype": "GGA_PBE_SOL",
        "vkloff": [0, 0, 0],
        "tforce": True,
        "nosource": False
    }
    ase.io.exciting.write_input_xml_file(
        file_name=file_path,
        atoms=nitrogen_trioxide_atoms,
        ground_state_input=ground_state_input_dict,
        species_path="/dummy/arbitrary/path",
        title=None)
    atoms_obj = ase.io.exciting.ase_atoms_from_exciting_input_xml(file_path)

    assert all(atoms_obj.symbols == "NOOO")
    assert atoms_obj.cell.array == pytest.approx(expected_cell)
    assert atoms_obj.positions == pytest.approx(expected_positions)


def test_parse_info_out_xml_bad_path(tmp_path, excitingtools):
    """Tests parse method raises error when info.out file doesn't exist."""
    output_file_path = Path(tmp_path).joinpath('info.out')

    with pytest.raises(FileNotFoundError):
        ase.io.exciting.parse_output(
            output_file_path)


def test_parse_info_out_energy(tmp_path, excitingtools):
    """Test parsing the INFO.OUT output from exciting using parse_output()."""
    expected_lattice_cell = [
        [10.3360193975, 10.3426010725, 0.0054547264],
        [-10.3461511392, 10.3527307290, 0.0059928210],
        [10.3354645037, 10.3540072605, 20.6246241525]]

    file = tmp_path / "INFO.OUT"
    file.write_text(LDA_VWN_AR_INFO_OUT)
    assert file.exists(), "INFO.OUT written to tmp_path"

    results = ase.io.exciting.parse_output(file)
    initialization = results['initialization']

    # Finally ensure that we that the final SCL cycle is what we expect and
    # the final SCL results can be accessed correctly:
    final_scl_iteration = list(results["scl"].keys())[-1]
    assert pytest.approx(
        float(results["scl"][final_scl_iteration][
            "Hartree energy"])) == 205.65454603
    assert pytest.approx(
        float(results["scl"][final_scl_iteration][
            "Estimated fundamental gap"])) == 0.36095838
    assert pytest.approx(float(results["scl"][
        final_scl_iteration]["Hartree energy"])) == 205.65454603
    assert pytest.approx(float(
        initialization['Unit cell volume'])) == 4412.7512103067

    # This used to be '1' (str) in version 'nitrogen' but is 1 (int)
    # as of version 'neon':
    assert int(initialization['Total number of k-points']) == 1
    assert int(initialization['Maximum number of plane-waves']) == 251

    # Grab the lattice vectors. excitingtools parses them in a fortran like
    # vector. We reshape accordingly into a 3x3 matrix where rows correspond
    # to lattice vectors.
    lattice_vectors_as_matrix = np.array(np.reshape(
        initialization['Lattice vectors (cartesian)'], (3, 3), 'F'), float)
    assert lattice_vectors_as_matrix.tolist() == expected_lattice_cell
