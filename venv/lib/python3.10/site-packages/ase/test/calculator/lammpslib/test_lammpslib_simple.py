import numpy as np
import pytest

import ase.io
from ase import Atom, Atoms, units
from ase.build import bulk
from ase.md.verlet import VelocityVerlet


@pytest.fixture()
def atoms_fcc_Ni_with_H_at_center():
    atoms = bulk("Ni", cubic=True)
    atoms += Atom("H", position=atoms.cell.diagonal() / 2)
    return atoms


@pytest.fixture()
def lammps_data_file_Fe(datadir):
    return datadir / "lammpslib_simple_input.data"


@pytest.fixture()
def calc_params_Fe(lammps_data_file_Fe):
    calc_params = {}
    calc_params["lammps_header"] = [
        "units           real",
        "atom_style      full",
        "boundary        p p p",
        "box tilt        large",
        "pair_style      lj/cut/coul/long 12.500",
        "bond_style      harmonic",
        "angle_style     harmonic",
        "kspace_style    ewald 0.0001",
        "kspace_modify   gewald 0.01",
        f"read_data      {lammps_data_file_Fe}",
    ]
    calc_params["lmpcmds"] = []
    calc_params["atom_types"] = {"Fe": 1}
    calc_params["create_atoms"] = False
    calc_params["create_box"] = False
    calc_params["boundary"] = False
    calc_params["log_file"] = "test.log"
    return calc_params


@pytest.fixture()
def atoms_Fe(lammps_data_file_Fe):
    return ase.io.read(
        lammps_data_file_Fe,
        format="lammps-data",
        Z_of_type={1: 26},
        units="real",
    )


@pytest.mark.calculator_lite()
@pytest.mark.calculator("lammpslib")
def test_lammpslib_simple(
    factory,
    calc_params_NiH,
    atoms_fcc_Ni_with_H_at_center: Atoms,
):
    # TODO: Split the test in a rational way.
    NiH = atoms_fcc_Ni_with_H_at_center

    # Add a bit of distortion to the cell
    NiH.set_cell(
        NiH.cell + [[0.1, 0.2, 0.4], [0.3, 0.2, 0.0], [0.1, 0.1, 0.1]],
        scale_atoms=True,
    )

    calc = factory.calc(**calc_params_NiH)
    NiH.calc = calc

    E = NiH.get_potential_energy()
    F = NiH.get_forces()
    S = NiH.get_stress()

    print("Energy: ", E)
    print("Forces:", F)
    print("Stress: ", S)
    print()

    E = NiH.get_potential_energy()
    F = NiH.get_forces()
    S = NiH.get_stress()

    calc = factory.calc(**calc_params_NiH)
    NiH.calc = calc

    E2 = NiH.get_potential_energy()
    F2 = NiH.get_forces()
    S2 = NiH.get_stress()

    assert E == pytest.approx(E2, rel=1e-4)
    assert F == pytest.approx(F2, rel=1e-4)
    assert S == pytest.approx(S2, rel=1e-4)

    NiH.rattle(stdev=0.2)
    E3 = NiH.get_potential_energy()
    F3 = NiH.get_forces()
    S3 = NiH.get_stress()

    print("rattled atoms")
    print("Energy: ", E3)
    print("Forces:", F3)
    print("Stress: ", S3)
    print()

    assert not np.allclose(E, E3)
    assert not np.allclose(F, F3)
    assert not np.allclose(S, S3)

    # Add another H
    NiH += Atom("H", position=NiH.cell.diagonal() / 4)
    E4 = NiH.get_potential_energy()
    F4 = NiH.get_forces()
    S4 = NiH.get_stress()

    assert not np.allclose(E4, E3)
    assert not np.allclose(F4[:-1, :], F3)
    assert not np.allclose(S4, S3)

    # the example from the docstring

    NiH = atoms_fcc_Ni_with_H_at_center
    calc = factory.calc(**calc_params_NiH)
    NiH.calc = calc
    print("Energy ", NiH.get_potential_energy())


@pytest.mark.parametrize("keep_alive", [False, True])
@pytest.mark.calculator_lite()
@pytest.mark.calculator("lammpslib")
def test_read_data(
    factory,
    calc_params_Fe,
    atoms_Fe: Atoms,
    keep_alive: bool,
):
    """Test `read_data` and `keep_alive`.

    Test if a LAMMPS data file can be read and if both `keep_alive=False` and
    `keep_alive=True` work together with `Dynamics`.

    Get energy from a LAMMPS calculation of an uncharged system.
    This was written to run with the 30 Apr 2019 version of LAMMPS,
    for which uncharged systems require the use of 'kspace_modify gewald'.
    """
    calc = factory.calc(keep_alive=keep_alive, **calc_params_Fe)
    atoms_Fe.calc = calc
    with VelocityVerlet(atoms_Fe, 1 * units.fs) as dyn:
        energy = atoms_Fe.get_potential_energy()
        assert energy == pytest.approx(2041.411982950972, rel=1e-4)

        dyn.run(10)
        energy = atoms_Fe.get_potential_energy()
        assert energy == pytest.approx(312.4315854721744, rel=1e-4)


@pytest.mark.calculator_lite()
@pytest.mark.calculator('lammpslib')
def test_charges_atomic(
    factory,
    calc_params_NiH: dict,
    atoms_fcc_Ni_with_H_at_center: Atoms,
):
    """Test charges for `atom_style atomic`.

    Since `atom_style atomic` does not include atomic charges, LAMMPS raises
    an error "Cannot set attribute charge for atom style atomic" if we set a
    command like 'set atom 1 charge 1.0'.
    The above command must therefore be set only when `atom_style` include
    atomic charges, and otherwise the command should be skipped.
    This test checks if the skipping is properly done.
    """
    atoms = atoms_fcc_Ni_with_H_at_center
    atoms.set_initial_charges(len(atoms) * [1.0])
    atoms.calc = factory.calc(**calc_params_NiH)
    atoms.get_potential_energy()


@pytest.mark.calculator_lite()
@pytest.mark.calculator('lammpslib')
def test_charges_full(
    factory,
    calc_params_Fe: dict,
    atoms_Fe: Atoms,
):
    """Test charges for `atom_style full`.

    Test if

    1. Setting initial charges triggers a new calculation.
    2. The energy with charges is different from the energy without charges.

    """
    calc = factory.calc(**calc_params_Fe)
    atoms_Fe.calc = calc

    energy_without_charges = atoms_Fe.get_potential_energy()

    atoms_Fe.set_initial_charges(len(atoms_Fe) * [1.0])

    energy_with_charges = atoms_Fe.get_potential_energy()

    assert energy_with_charges != pytest.approx(energy_without_charges)
