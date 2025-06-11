from pathlib import Path

import numpy as np
import pytest

import ase.io
from ase import Atoms
from ase.build import bulk
from ase.calculators.tersoff import Tersoff, TersoffParameters
from ase.units import bar


def read_lammps_dump_data(dump_file: Path) -> dict:
    """Read energy, pressure tensor and forces from a custom LAMMPS dump file.

    The dump file must contain comments with the total energy and pressure
    components. The file must also contain the standard LAMMPS dump format.

    The total energy is read from a comment line of the form
    `# Total Energy (eV): <energy>`. The pressure components are read from a
    comment line of the form
    `# Pressure (bars): <pxx> <pyy> <pzz> <pxy> <pxz> <pyz>`.

    """
    data: dict = {
        'total_energy': None,
        'pressure_tensor': None,
        'forces': [],
        'atoms': None,
    }

    with open(dump_file, 'r') as file:
        in_atoms_section: bool = False
        for line in file:
            if line.startswith('# Total Energy'):
                data['total_energy'] = float(line.split(':')[-1])
            elif line.startswith('# Pressure'):
                # Parse pressure tensor components (xx, yy, zz, xy, xz, yz)
                pressure_components = [
                    float(x) for x in line.split(':')[-1].split()
                ]
                data['pressure_tensor'] = np.array(pressure_components)
            elif line.startswith('ITEM: ATOMS'):
                in_atoms_section = True
                continue
            elif in_atoms_section:
                parts = line.strip().split()
                data['forces'].append(
                    [float(parts[5]), float(parts[6]), float(parts[7])]
                )

    data['forces'] = np.array(data['forces'])
    return data


def validate_results(atoms: Atoms, reference_data: dict):
    """Validate calculated results against reference data."""
    # Energy validation
    calc_energy = atoms.get_potential_energy()
    abs_ref_energy = abs(reference_data['total_energy'])
    rel_energy_diff = (
        abs(calc_energy - reference_data['total_energy']) / abs_ref_energy
    )

    assert rel_energy_diff < 1.0e-5, (
        f"Total energy mismatch: calculated={calc_energy:.6f} eV, "
        f"reference={reference_data['total_energy']:.6f} eV, "
        f"relative difference={rel_energy_diff:.6f}"
    )
    # Forces validation
    calculated_forces = atoms.get_forces()
    np.testing.assert_allclose(
        calculated_forces,
        reference_data['forces'],
        atol=1.0e-4,
        err_msg='Forces mismatch',
    )

    # Stress tensor validation
    if reference_data['pressure_tensor'] is not None:
        # Get stress from ASE (returned as [xx, yy, zz, yz, xz, xy])
        calculated_stress = atoms.get_stress()
        # Convert to bar and use positive convention
        calculated_pressure = calculated_stress / bar
        calculated_pressure_reordered = np.array(
            [
                calculated_pressure[0],  # xx
                calculated_pressure[1],  # yy
                calculated_pressure[2],  # zz
                calculated_pressure[5],  # xy
                calculated_pressure[4],  # xz
                calculated_pressure[3],  # yz
            ]
        )

        np.testing.assert_allclose(
            calculated_pressure_reordered,
            reference_data['pressure_tensor'],
            rtol=1.0e-5,
            atol=1e-8,
            err_msg='Pressure tensor mismatch',
        )


@pytest.fixture
def si_parameters():
    """Fixture providing the Silicon parameters.

    Parameters taken from: Tersoff, Phys Rev B, 37, 6991 (1988)
    """
    return {
        ('Si', 'Si', 'Si'): TersoffParameters(
            A=3264.7,
            B=95.373,
            lambda1=3.2394,
            lambda2=1.3258,
            lambda3=1.3258,
            beta=0.33675,
            gamma=1.00,
            m=3.00,
            n=22.956,
            c=4.8381,
            d=2.0417,
            h=0.0000,
            R=3.00,
            D=0.20,
        )
    }


def test_initialize_from_params_from_dict(si_parameters):
    """Test initializing Tersoff calculator from dictionary of parameters."""
    calc = Tersoff(si_parameters)
    assert calc.parameters == si_parameters
    diamond = bulk('Si', 'diamond', a=5.43)
    diamond.calc = calc
    potential_energy = diamond.get_potential_energy()
    np.testing.assert_allclose(potential_energy, -9.260818674314585, atol=1e-8)


def test_set_parameters(si_parameters):
    """Test updating parameters of the Tersoff calculator."""
    calc = Tersoff(si_parameters)
    key = ('Si', 'Si', 'Si')

    calc.set_parameters(key, m=2.0)
    assert calc.parameters[key].m == 2.0

    calc.set_parameters(key, R=2.90, D=0.25)
    assert calc.parameters[key].R == 2.90
    assert calc.parameters[key].D == 0.25

    new_params = TersoffParameters(
        m=si_parameters[key].m,
        gamma=si_parameters[key].gamma,
        lambda3=si_parameters[key].lambda3,
        c=si_parameters[key].c,
        d=si_parameters[key].d,
        h=si_parameters[key].h,
        n=si_parameters[key].n,
        beta=si_parameters[key].beta,
        lambda2=si_parameters[key].lambda2,
        B=si_parameters[key].B,
        R=3.00,  # Reset cutoff radius
        D=si_parameters[key].D,
        lambda1=si_parameters[key].lambda1,
        A=si_parameters[key].A,
    )
    calc.set_parameters(key, params=new_params)
    assert calc.parameters[key] == new_params


@pytest.mark.parametrize('system', ['silicon', 'silicon_carbide'])
def test_tersoff(datadir, system):
    """
    Test the Tersoff potential for various systems.

    This test checks the correctness of the Tersoff interatomic potential
    implementation by comparing calculated energy, forces, and stress tensor
    values against reference LAMMPS data for given systems.

    The test ensures that the Tersoff potential produces results within the
    acceptable tolerance levels for each system.
    """
    # Load the test data files
    dump_file = datadir / f'tersoff/dump.{system}'
    potential_file = datadir / f'tersoff/{system}.tersoff'

    assert dump_file.exists(), f'Dump file not found: {dump_file}'
    assert potential_file.exists(), f"""
    Potential file not found: {potential_file}
    """

    atoms = ase.io.read(dump_file, format='lammps-dump-text')
    reference_data = read_lammps_dump_data(dump_file)

    calc = Tersoff.from_lammps(str(potential_file))
    atoms.calc = calc

    validate_results(atoms, reference_data)
