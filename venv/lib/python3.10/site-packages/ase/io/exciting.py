# fmt: off

"""This is the implementation of the exciting I/O functions.

The main roles these functions do is write exciting ground state
input files and read exciting ground state ouput files.

Right now these functions all written without a class to wrap them. This
could change in the future but was done to make things simpler.

These functions are primarily called by the exciting caculator in
ase/calculators/exciting/exciting.py.

See the correpsonding test file in ase/test/io/test_exciting.py.

Plan is to add parsing of eigenvalues in the next iteration using
excitingtools.exciting_dict_parsers.groundstate_parser.parse_eigval

Note: excitingtools must be installed using `pip install excitingtools` for
the exciting io to work.
"""
from pathlib import Path
from typing import Dict, Optional, Union

import ase


def parse_output(info_out_file_path):
    """Parse exciting INFO.OUT output file using excitingtools.

    Note, excitingtools works by returning a dictionary that contains
    two high level keys. Initialization and results. Initialization
    contains data about how the calculation was setup (e.g. structure,
    maximum number of planewaves, etc...) and the results
    gives SCF cycle result information (e.g. total energy).

    Args:
        info_out_file_path: path to an INFO.out exciting output file.
    Returns:
        A dictionary containing information about how the calculation was setup
        and results from the calculations SCF cycles.
    """
    from excitingtools.exciting_dict_parsers.groundstate_parser import (
        parse_info_out,
    )

    # Check for the file:
    if not Path(info_out_file_path).is_file():
        raise FileNotFoundError
    return parse_info_out(info_out_file_path)


def write_input_xml_file(
        file_name, atoms: ase.Atoms, ground_state_input: Dict,
        species_path, title=None,
        properties_input: Optional[Dict] = None):
    """Write input xml file for exciting calculation.

    Args:
        file_name: where to save the input xml file.
        atoms: ASE Atoms object.
        ground_state_input: ground state parameters for run.
        properties_input: optional additional parameters to run
            after performing the ground state calculation (e.g. bandstructure
            or DOS.)
    """
    from excitingtools import (
        ExcitingGroundStateInput,
        ExcitingInputXML,
        ExcitingPropertiesInput,
        ExcitingStructure,
    )

    # Convert ground state dictionary into expected input object.
    ground_state = ExcitingGroundStateInput(**ground_state_input)
    structure = ExcitingStructure(atoms, species_path=species_path)
    # If we are running futher calculations such as bandstructure/DOS.
    if properties_input is not None:
        properties_input = ExcitingPropertiesInput(**properties_input)
    else:
        properties_input = ExcitingPropertiesInput()
    input_xml = ExcitingInputXML(structure=structure,
                                 groundstate=ground_state,
                                 properties=properties_input,
                                 title=title)

    input_xml.write(file_name)


def ase_atoms_from_exciting_input_xml(
        input_xml_path: Union[Path, str]) -> ase.Atoms:
    """Helper function to read structure from input.xml file.

    Note, this function operates on the input.xml file that is the input
    to an exciting calculation. It parses the structure data given in the file
    and returns it in an ase Atoms object. Note this information can also be
    taken from an INFO.out file using parse_output. This script is more
    lightweight than parse_output since the input xml is significantly smaller
    than an INFO.out file and is XML structured making the parsing easier.

    Args:
        input_xml_path: Path where input.xml file lives.

    Returns:
        ASE atoms object with all the relevant fields filled.
    """
    from excitingtools.exciting_obj_parsers.input_xml import parse_input_xml
    from excitingtools.structure.ase_utilities import exciting_structure_to_ase
    structure = parse_input_xml(input_xml_path).structure
    return exciting_structure_to_ase(structure)
