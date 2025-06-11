import pytest

from ase.atoms import Atoms


@pytest.mark.calculator_lite()
@pytest.mark.calculator('lammpsrun')
def test_no_data_file_wrap(factory):
    """
    If 'create_atoms' hasn't been given the appropriate 'remap yes' option,
    atoms falling outside of a periodic cell are not actually created.  The
    lammpsrun calculator will then look at the thermo output and determine a
    discrepancy the number of atoms reported compared to the length of the
    ASE Atoms object and raise a RuntimeError.  This problem can only
    possibly arise when the 'no_data_file' option for the calculator is set
    to True.  Furthermore, note that if atoms fall outside of the box along
    non-periodic dimensions, create_atoms is going to refuse to create them
    no matter what, so you simply can't use the 'no_data_file' option if you
    want to allow for that scenario.
    """

    # Make a periodic box and put one atom outside of it
    pos = [[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]
    atoms = Atoms(symbols=["Ar"] * 2, positions=pos, cell=[10.0, 10.0, 10.0],
                  pbc=True)

    # Set parameters for calculator
    params = {}
    params["pair_style"] = "lj/cut 8.0"
    params["pair_coeff"] = ["1 1 0.0108102 3.345"]

    # Don't write a data file string. This will force
    # ase.calculators.lammps.inputwriter.write_lammps_in to write a bunch of
    # 'create_atoms' commands into the LAMMPS input file
    params["no_data_file"] = True

    with factory.calc(specorder=["Ar"], **params) as calc:
        atoms.calc = calc
        atoms.get_potential_energy()
        # assert something?
