"""
The OPLS io module uses the coordinatetransform.py module under
calculators/lammps/.  This test simply ensures that it uses that module
correctly.  First, it ensures that the interface there hasn't changed.  Second,
it ensures that coordinate system changes are accounted for correctly by
starting with a structure that does not meet LAMMPS' convention for cell
orientation and checking that it gets rotated into the expected configuration
"""
import numpy as np
import pytest

from ase.io.opls import OPLSff, OPLSStructure


@pytest.fixture()
def opls_structure_file_name(datadir):
    return datadir / "opls_structure_ext.xyz"


@pytest.fixture()
def opls_force_field_file_name(datadir):
    """Need to define OPLS parameters for each species in order to be able to
    write a lammps input file"""
    # TODO: This parameter file is being read from the 'testdata' directory,
    #       but it actually already exists under doc/ase/io.  Any sensible way
    #       to prevent duplicating information?
    return str(datadir / "172_defs.par")


def test_opls_write_lammps(opls_structure_file_name,
                           opls_force_field_file_name):

    LAMMPS_FILES_PREFIX = "lmp"

    # Get structure
    atoms = OPLSStructure(opls_structure_file_name)

    # Set up force field object
    with open(opls_force_field_file_name) as fd:
        opls_force_field = OPLSff(fd)

    # Write input files for lammps to current directory
    opls_force_field.write_lammps(atoms, prefix=LAMMPS_FILES_PREFIX)

    # Read the lammps data file
    with open(LAMMPS_FILES_PREFIX + "_atoms") as fd:
        lammps_data = fd.readlines()

    # Locate Atoms block and extract the data for the three atoms in the
    # input structure
    for ind, line in enumerate(lammps_data):
        if line.startswith("Atoms"):
            atom1_data = lammps_data[ind + 2]
            atom2_data = lammps_data[ind + 3]
            atom3_data = lammps_data[ind + 4]
            break

    # Grab positions from data
    pos_indices = slice(4, 7)
    atom1_pos = np.array(atom1_data.split()[pos_indices], dtype=float)
    atom2_pos = np.array(atom2_data.split()[pos_indices], dtype=float)
    atom3_pos = np.array(atom3_data.split()[pos_indices], dtype=float)

    # Check that positions match expected values
    assert atom1_pos == pytest.approx(
        np.array([1.6139, -0.7621, 0.0]), abs=1e-4)
    assert atom2_pos == pytest.approx(np.array([-0.3279, 0.5227, 0]), abs=1e-4)
    assert atom3_pos == pytest.approx(
        np.array([-0.96, 0.5809, 0.88750]), abs=1e-4)
