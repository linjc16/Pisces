# type: ignore
import numpy as np

import ase.io.rmc6f as rmc6f
from ase import Atoms
from ase.io import read, write
from ase.lattice.compounds import TRI_Fe2O3

rmc6f_input_text = """
(Version 6f format configuration file)
Metadata owner:     Joe Smith
Metadata date:      31-01-1900
Metadata material:  SF6
Metadata comment:  Some comments go here...
Metadata source:    GSAS
Number of moves generated:           89692
Number of moves tried:               85650
Number of moves accepted:            10074
Number of prior configuration saves: 0
Number of atoms:                     7
Number density (Ang^-3):                 0.068606
Supercell dimensions:                1 1 1
Cell (Ang/deg): 4.672816 4.672816 4.672816 90.0 90.0 90.0
Lattice vectors (Ang):
    4.672816    0.000000    0.000000
    0.000000    4.672816    0.000000
    0.000000    0.000000    4.672816
Atoms:
     1   S     0.600452    0.525100    0.442050     1   0   0   0
     2   F     0.911952    0.450722    0.382733     2   0   0   0
     3   F     0.283794    0.616712    0.500094     3   0   0   0
     4   F     0.679823    0.854839    0.343915     4   0   0   0
     5   F     0.531660    0.229024    0.535688     5   0   0   0
     6   F     0.692514    0.584931    0.746683     6   0   0   0
     7   F     0.509687    0.449350    0.111960     7   0   0   0
"""
# Information for system
positions = [
    [0.600452, 0.525100, 0.442050],
    [0.911952, 0.450722, 0.382733],
    [0.283794, 0.616712, 0.500094],
    [0.679823, 0.854839, 0.343915],
    [0.531660, 0.229024, 0.535688],
    [0.692514, 0.584931, 0.746683],
    [0.509687, 0.449350, 0.111960]]
symbols = ['S', 'F', 'F', 'F', 'F', 'F', 'F']
lat = 4.672816

# Intermediate data structures to test against
symbol_xyz_list = [[s] + xyz for s, xyz in zip(symbols, positions)]
symbol_xyz_dict = {i + 1: row for i, row in enumerate(symbol_xyz_list)}
symbol_xyz_list_ext = [tuple([i + 1] + row + [0, 0, 0, 0])
                       for i, row in enumerate(symbol_xyz_list)]

# Target Atoms system
lat_positions = lat * np.array(positions)
cell = [lat, lat, lat]
rmc6f_atoms = Atoms(symbols, positions=lat_positions, cell=cell, pbc=[1, 1, 1])


def test_rmc6f_read():
    """Test for reading rmc6f input file."""
    with open('input.rmc6f', 'w') as rmc6f_input_f:
        rmc6f_input_f.write(rmc6f_input_text)

    rmc6f_input_atoms = read('input.rmc6f')
    assert len(rmc6f_input_atoms) == 7
    assert rmc6f_input_atoms == rmc6f_atoms


def test_rmc6f_write():
    """Test for writing rmc6f input file."""
    tol = 1e-5
    write('output.rmc6f', rmc6f_atoms)
    readback = read('output.rmc6f')
    assert np.allclose(rmc6f_atoms.positions, readback.positions, rtol=tol)
    assert readback == rmc6f_atoms


def test_rmc6f_write_with_order():
    """Test for writing rmc6f input file with order passed in."""
    tol = 1e-5
    write('output.rmc6f', rmc6f_atoms, order=['F', 'S'])
    readback = read('output.rmc6f')
    reordered_positions = np.vstack(
        (rmc6f_atoms.positions[1:7], rmc6f_atoms.positions[0]))
    assert np.allclose(reordered_positions, readback.positions, rtol=tol)


def test_rmc6f_write_with_triclinic_system():
    """Test for writing rmc6f input file for triclinic system
    """
    tol = 1e-5
    fe4o6 = TRI_Fe2O3(
        symbol=('Fe', 'O'),
        latticeconstant={
            'a': 5.143,
            'b': 5.383,
            'c': 14.902,
            'alpha': 90.391,
            'beta': 90.014,
            'gamma': 89.834},
        size=(1, 1, 1))

    # Make sure these are the returned cells (verified correct for rmc6f file)
    va = [5.143, 0.0, 0.0]
    vb = [0.015596, 5.382977, 0.0]
    vc = [-0.00364124, -0.101684, 14.901653]

    write('output.rmc6f', fe4o6)
    readback = read('output.rmc6f')
    assert np.allclose(fe4o6.positions, readback.positions, rtol=tol)
    assert np.allclose(va, readback.cell[0], rtol=tol)
    assert np.allclose(vb, readback.cell[1], rtol=tol)
    assert np.allclose(vc, readback.cell[2], rtol=tol)


def test_rmc6f_read_construct_regex():
    """Test for utility function that constructs rmc6f header regex."""
    header_lines = [
        "Number of atoms:",
        "  Supercell dimensions:  ",
        "    Cell (Ang/deg):  ",
        "      Lattice vectors (Ang):  "]
    result = rmc6f._read_construct_regex(header_lines)
    target = "(Number\\s+of\\s+atoms:|Supercell\\s+dimensions:|Cell\\s+\\(Ang/deg\\):|Lattice\\s+vectors\\s+\\(Ang\\):)"  # noqa: E501
    assert result == target


def test_rmc6f_read_line_of_atoms_section_style_no_labels():
    """Test for reading a line of atoms section
    w/ 'no labels' style for rmc6f
    """
    atom_line = "1 S 0.600452 0.525100 0.442050 1 0 0 0"
    atom_id, props = rmc6f._read_line_of_atoms_section(atom_line.split())
    target_id = 1
    target_props = ['S', 0.600452, 0.5251, 0.44205]
    assert atom_id == target_id
    assert props == target_props


def test_rmc6f_read_line_of_atoms_section_style_labels():
    """Test for reading a line of atoms section
    w/ 'labels'-included style for rmc6f
    """
    atom_line = "1 S [1] 0.600452 0.525100 0.442050 1 0 0 0"
    atom_id, props = rmc6f._read_line_of_atoms_section(atom_line.split())
    target_id = 1
    target_props = ['S', 0.600452, 0.5251, 0.44205]
    assert atom_id == target_id
    assert props == target_props


def test_rmc6f_read_line_of_atoms_section_style_magnetic():
    """Test for reading a line of atoms section
    w/ 'magnetic' style for rmc6f
    """
    atom_line = "1 S 0.600452 0.525100 0.442050 1 0 0 0 M: 0.1"
    atom_id, props = rmc6f._read_line_of_atoms_section(atom_line.split())
    target_id = 1
    target_props = ['S', 0.600452, 0.5251, 0.44205, 0.1]
    assert atom_id == target_id
    assert props == target_props


def test_rmc6f_read_process_rmc6f_lines_to_pos_and_cell():
    """Test for utility function that processes lines of rmc6f using
    regular expressions to capture atom properties and cell information
    """
    tol = 1e-5
    lines = rmc6f_input_text.split('\n')
    props, cell = rmc6f._read_process_rmc6f_lines_to_pos_and_cell(lines)

    target_cell = np.zeros((3, 3), float)
    np.fill_diagonal(target_cell, 4.672816)

    assert props == symbol_xyz_dict
    assert np.allclose(cell, target_cell, rtol=tol)


def test_rmc6f_read_process_rmc6f_lines_to_pos_and_cell_padded_whitespace():
    """Test for utility function that processes lines of rmc6f using
    regular expressions to capture atom properties and cell information
    with puposeful whitespace padded on one line
    """
    tol = 1e-5
    lines = rmc6f_input_text.split('\n')
    lines[14] = f"    {lines[14]}    "  # intentional whitespace
    props, cell = rmc6f._read_process_rmc6f_lines_to_pos_and_cell(lines)

    target_cell = np.zeros((3, 3), float)
    np.fill_diagonal(target_cell, 4.672816)

    assert props == symbol_xyz_dict
    assert np.allclose(cell, target_cell, rtol=tol)


def test_rmc6f_write_output_column_format():
    """Test for utility function that processes the columns in array
    and gets back out formatting information.
    """
    cols = ['id', 'symbols', 'scaled_positions', 'ref_num', 'ref_cell']
    arrays = {}
    arrays['id'] = np.array([1, 2, 3, 4, 5, 6, 7])
    arrays['symbols'] = np.array(symbols)
    arrays['ref_num'] = np.zeros(7, int)
    arrays['ref_cell'] = np.zeros((7, 3), int)
    arrays['scaled_positions'] = lat_positions / lat

    ncols, _dtype_obj, fmt = rmc6f._write_output_column_format(cols, arrays)

    target_ncols = [1, 1, 3, 1, 3]
    target_fmt = "%8d %s%14.6f %14.6f %14.6f %8d %8d %8d %8d \n"

    assert ncols == target_ncols
    assert fmt == target_fmt


def test_rmc6f_write_output():
    """Test for utility function for writing rmc6f output
    """
    header_lines = [
        '(Version 6f format configuration file)',
        '(Generated by ASE - Atomic Simulation Environment https://wiki.fysik.dtu.dk/ase/ )',  # noqa: E501
        "Metadata date:18-007-'2019",
        'Number of types of atoms:   2 ',
        'Atom types present:          S F',
        'Number of each atom type:   1 6',
        'Number of moves generated:           0',
        'Number of moves tried:               0',
        'Number of moves accepted:            0',
        'Number of prior configuration saves: 0',
        'Number of atoms:                     7',
        'Supercell dimensions:                1 1 1',
        'Number density (Ang^-3):              0.06860598423060468',
        'Cell (Ang/deg): 4.672816 4.672816 4.672816 90.0 90.0 90.0',
        'Lattice vectors (Ang):',
        '    4.672816     0.000000     0.000000',
        '    0.000000     4.672816     0.000000',
        '    0.000000     0.000000     4.672816',
        'Atoms:']

    data_array = symbol_xyz_list_ext
    data_dtype = [
        ('id', '<i8'),
        ('symbols', '<U1'),
        ('scaled_positions0', '<f8'),
        ('scaled_positions1', '<f8'),
        ('scaled_positions2', '<f8'),
        ('ref_num', '<i8'),
        ('ref_cell0', '<i8'),
        ('ref_cell1', '<i8'),
        ('ref_cell2', '<i8')]
    data = np.array(data_array, dtype=data_dtype)
    fmt = "%8d %s%14.6f %14.6f %14.6f %8d %8d %8d %8d \n"
    with open('output.rmc6f', 'w') as fileobj:
        rmc6f._write_output(fileobj, header_lines, data, fmt)
