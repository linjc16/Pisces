import os
from io import StringIO

import ase.io as aio
from ase.io.gromacs import read_gromacs, write_gromacs


def test_gromacs_1():
    standard_gromacs = '''MD of 2 waters, t= 0.0
    6
    1WATER  OW1    1   0.126   1.624   1.679  0.1227 -0.0580  0.0434
    1WATER  HW2    2   0.190   1.661   1.747  0.8085  0.3191 -0.7791
    1WATER  HW3    3   0.177   1.568   1.613 -0.9045 -2.6469  1.3180
    2WATER  OW1    4   1.275   0.053   0.622  0.2519  0.3140 -0.1734
    2WATER  HW2    5   1.337   0.002   0.680 -1.0641 -1.1349  0.0257
    2WATER  HW3    6   1.326   0.120   0.568  1.9427 -0.8216 -0.0244
   1.82060   1.82060   1.82060
'''
    # read x from StringIO
    gromacs_file = StringIO(standard_gromacs)
    x = read_gromacs(gromacs_file)

    # write x to StringIO
    out_stringio = StringIO()
    write_gromacs(out_stringio, x)
    out_stringio.seek(0)

    # verify
    output = out_stringio.read()
    ase_output = '\n'.join(output.split('\n')[1:-2])
    standard_output = '\n'.join(standard_gromacs.split('\n')[1:-2])
    # import pdb; pdb.set_trace()
    assert (ase_output == standard_output
            ), 'ASE gromacs output differs from standard output'


def test_gromacs_2(tmp_path: str):
    water_str = """water
    3
    1SOL  OW000    1   0.000   0.000   0.000
    1SOL  HW001    2   0.000   0.100   0.000
    1SOL  HW002    3   0.088  -0.036   0.000
   1.00000   1.00000   1.00000\n"""
    gro_file = os.path.join(tmp_path, "water.gro")
    with open(gro_file, "w") as f:
        f.write(water_str)

    ase_atoms = read_gromacs(gro_file)
    assert str(ase_atoms.symbols) != "W3"

    byte_atoms = read_gromacs(gro_file)
    assert str(byte_atoms.symbols) == "OH2"

    dump_gro_file = os.path.join(tmp_path, "dump_water.gro")
    aio.write(dump_gro_file, byte_atoms)
    reload_atoms = read_gromacs(dump_gro_file)
    for atom_a, atom_b in zip(byte_atoms, reload_atoms):
        assert atom_a.symbol == atom_b.symbol
        assert all(atom_a.position == atom_b.position)
