# fmt: off

import io
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixCartesian, FixConstraint
from ase.io import ParseError
from ase.utils import reader, string2index


@reader
def read_castep_castep(fd, index=-1):
    """Read a .castep file and returns an Atoms object.

    The calculator information will be stored in the calc attribute.

    Notes
    -----
    This routine will return an atom ordering as found within the castep file.
    This means that the species will be ordered by ascending atomic numbers.
    The atoms witin a species are ordered as given in the original cell file.

    """
    # look for last result, if several CASTEP run are appended
    line_start, line_end, end_found = _find_last_record(fd)
    if not end_found:
        raise ParseError(f'No regular end found in {fd.name} file.')

    # These variables are finally assigned to `SinglePointCalculator`
    # for backward compatibility with the `Castep` calculator.
    cut_off_energy = None
    kpoints = None
    total_time = None
    peak_memory = None

    # jump back to the beginning to the last record
    fd.seek(0)
    for i, line in enumerate(fd):
        if i == line_start:
            break

    # read header
    parameters_header = _read_header(fd)
    if 'cut_off_energy' in parameters_header:
        cut_off_energy = parameters_header['cut_off_energy']
        if 'basis_precision' in parameters_header:
            del parameters_header['cut_off_energy']  # avoid conflict

    markers_new_iteration = [
        'BFGS: starting iteration',
        'BFGS: improving iteration',
        'Starting MD iteration',
    ]

    images = []

    results = {}
    constraints = []
    species_pot = []
    castep_warnings = []
    for i, line in enumerate(fd):

        if i > line_end:
            break

        if 'Number of kpoints used' in line:
            kpoints = int(line.split('=')[-1].strip())
        elif 'Unit Cell' in line:
            lattice_real = _read_unit_cell(fd)
        elif 'Cell Contents' in line:
            for line in fd:
                if 'Total number of ions in cell' in line:
                    n_atoms = int(line.split()[7])
                if 'Total number of species in cell' in line:
                    int(line.split()[7])
                fields = line.split()
                if len(fields) == 0:
                    break
        elif 'Fractional coordinates of atoms' in line:
            species, custom_species, positions_frac = \
                _read_fractional_coordinates(fd, n_atoms)
        elif 'Files used for pseudopotentials' in line:
            for line in fd:
                line = fd.readline()
                if 'Pseudopotential generated on-the-fly' in line:
                    continue
                fields = line.split()
                if len(fields) == 2:
                    species_pot.append(fields)
                else:
                    break
        elif 'k-Points For BZ Sampling' in line:
            # TODO: generalize for non-Monkhorst Pack case
            # (i.e. kpoint lists) -
            # kpoints_offset cannot be read this way and
            # is hence always set to None
            for line in fd:
                if not line.strip():
                    break
                if 'MP grid size for SCF calculation' in line:
                    # kpoints =  ' '.join(line.split()[-3:])
                    # self.kpoints_mp_grid = kpoints
                    # self.kpoints_mp_offset = '0. 0. 0.'
                    # not set here anymore because otherwise
                    # two calculator objects go out of sync
                    # after each calculation triggering unnecessary
                    # recalculation
                    break

        elif 'Final energy' in line:
            key = 'energy_without_dispersion_correction'
            results[key] = float(line.split()[-2])
        elif 'Final free energy' in line:
            key = 'free_energy_without_dispersion_correction'
            results[key] = float(line.split()[-2])
        elif 'NB est. 0K energy' in line:
            key = 'energy_zero_without_dispersion_correction'
            results[key] = float(line.split()[-2])

        # Add support for dispersion correction
        # filtering due to SEDC is done in get_potential_energy
        elif 'Dispersion corrected final energy' in line:
            key = 'energy_with_dispersion_correlation'
            results[key] = float(line.split()[-2])
        elif 'Dispersion corrected final free energy' in line:
            key = 'free_energy_with_dispersion_correlation'
            results[key] = float(line.split()[-2])
        elif 'NB dispersion corrected est. 0K energy' in line:
            key = 'energy_zero_with_dispersion_correlation'
            results[key] = float(line.split()[-2])

        # check if we had a finite basis set correction
        elif 'Total energy corrected for finite basis set' in line:
            key = 'energy_with_finite_basis_set_correction'
            results[key] = float(line.split()[-2])

        # ******************** Forces *********************
        # ************** Symmetrised Forces ***************
        # ******************** Constrained Forces ********************
        # ******************* Unconstrained Forces *******************
        elif re.search(r'\**.* Forces \**', line):
            forces, constraints = _read_forces(fd, n_atoms)
            results['forces'] = np.array(forces)

        # ***************** Stress Tensor *****************
        # *********** Symmetrised Stress Tensor ***********
        elif re.search(r'\**.* Stress Tensor \**', line):
            results.update(_read_stress(fd))

        elif any(_ in line for _ in markers_new_iteration):
            _add_atoms(
                images,
                lattice_real,
                species,
                custom_species,
                positions_frac,
                constraints,
                results,
            )
            # reset for the next step
            lattice_real = None
            species = None
            positions_frac = None
            results = {}
            constraints = []

        # extract info from the Mulliken analysis
        elif 'Atomic Populations' in line:
            results.update(_read_mulliken_charges(fd))

        # extract detailed Hirshfeld analysis (iprint > 1)
        elif 'Hirshfeld total electronic charge (e)' in line:
            results.update(_read_hirshfeld_details(fd, n_atoms))

        elif 'Hirshfeld Analysis' in line:
            results.update(_read_hirshfeld_charges(fd))

        # There is actually no good reason to get out of the loop
        # already at this point... or do I miss something?
        # elif 'BFGS: Final Configuration:' in line:
        #    break
        elif 'warn' in line.lower():
            castep_warnings.append(line)

        # fetch some last info
        elif 'Total time' in line:
            pattern = r'.*=\s*([\d\.]+) s'
            total_time = float(re.search(pattern, line).group(1))

        elif 'Peak Memory Use' in line:
            pattern = r'.*=\s*([\d]+) kB'
            peak_memory = int(re.search(pattern, line).group(1))

    # add the last image
    _add_atoms(
        images,
        lattice_real,
        species,
        custom_species,
        positions_frac,
        constraints,
        results,
    )

    for atoms in images:
        # these variables are temporarily assigned to `SinglePointCalculator`
        # to be assigned to the `Castep` calculator for backward compatibility
        atoms.calc._cut_off_energy = cut_off_energy
        atoms.calc._kpoints = kpoints
        atoms.calc._species_pot = species_pot
        atoms.calc._total_time = total_time
        atoms.calc._peak_memory = peak_memory
        atoms.calc._parameters_header = parameters_header

    if castep_warnings:
        warnings.warn('WARNING: .castep file contains warnings')
        for warning in castep_warnings:
            warnings.warn(warning)

    if isinstance(index, str):
        index = string2index(index)

    return images[index]


def _find_last_record(fd):
    """Find the last record of the .castep file.

    Returns
    -------
    start : int
        Line number of the first line of the last record.
    end : int
        Line number of the last line of the last record.
    end_found : bool
        True if the .castep file ends as expected.

    """
    start = -1
    for i, line in enumerate(fd):
        if (('Welcome' in line or 'Materials Studio' in line)
                and 'CASTEP' in line):
            start = i

    if start < 0:
        warnings.warn(
            f'Could not find CASTEP label in result file: {fd.name}.'
            ' Are you sure this is a .castep file?'
        )
        return None

    # search for regular end of file
    end_found = False
    # start to search from record beginning from the back
    # and see if
    end = -1
    fd.seek(0)
    for i, line in enumerate(fd):
        if i < start:
            continue
        if 'Finalisation time   =' in line:
            end_found = True
            end = i
            break

    return (start, end, end_found)


def _read_header(out: io.TextIOBase):
    """Read the header blocks from a .castep file.

    Returns
    -------
    parameters : dict
        Dictionary storing keys and values of a .param file.
    """
    def _parse_on_off(_: str):
        return {'on': True, 'off': False}[_]

    read_title = False
    parameters: Dict[str, Any] = {}
    for line in out:
        if len(line) == 0:  # end of file
            break
        if re.search(r'^\s*\*+$', line) and read_title:  # end of header
            break

        if re.search(r'\**.* Title \**', line):
            read_title = True

        # General Parameters

        elif 'output verbosity' in line:
            parameters['iprint'] = int(line.split()[-1][1])
        elif re.match(r'\stype of calculation\s*:', line):
            parameters['task'] = {
                'single point energy': 'SinglePoint',
                'geometry optimization': 'GeometryOptimization',
                'band structure': 'BandStructure',
                'molecular dynamics': 'MolecularDynamics',
                'optical properties': 'Optics',
                'phonon calculation': 'Phonon',
                'E-field calculation': 'Efield',
                'Phonon followed by E-field': 'Phonon+Efield',
                'transition state search': 'TransitionStateSearch',
                'Magnetic Resonance': 'MagRes',
                'Core level spectra': 'Elnes',
                'Electronic Spectroscopy': 'ElectronicSpectroscopy',
            }[line.split(':')[-1].strip()]
        elif 'stress calculation' in line:
            parameters['calculate_stress'] = _parse_on_off(line.split()[-1])
        elif 'calculation limited to maximum' in line:
            parameters['run_time'] = float(line.split()[-2])
        elif re.match(r'\soptimization strategy\s*:', line):
            parameters['opt_strategy'] = {
                'maximize speed(+++)': 'Speed',
                'minimize memory(---)': 'Memory',
                'balance speed and memory': 'Default',
            }[line.split(':')[-1].strip()]

        # Exchange-Correlation Parameters
        elif re.match(r'\susing functional\s*:', line):
            functional_abbrevs = {
                'Local Density Approximation': 'LDA',
                'Perdew Wang (1991)': 'PW91',
                'Perdew Burke Ernzerhof': 'PBE',
                'revised Perdew Burke Ernzerhof': 'RPBE',
                'PBE with Wu-Cohen exchange': 'WC',
                'PBE for solids (2008)': 'PBESOL',
                'Hartree-Fock': 'HF',
                'Hartree-Fock +': 'HF-LDA',
                'Screened Hartree-Fock': 'sX',
                'Screened Hartree-Fock + ': 'sX-LDA',
                'hybrid PBE0': 'PBE0',
                'hybrid B3LYP': 'B3LYP',
                'hybrid HSE03': 'HSE03',
                'hybrid HSE06': 'HSE06',
                'RSCAN': 'RSCAN',
            }

            # If the name is not recognised, use the whole string.
            # This won't work in a new calculation, so will need to load from
            # .param file in such cases... but at least it will fail rather
            # than use the wrong XC!
            _xc_full_name = line.split(':')[-1].strip()
            parameters['xc_functional'] = functional_abbrevs.get(
                _xc_full_name, _xc_full_name)

        elif 'DFT+D: Semi-empirical dispersion correction' in line:
            parameters['sedc_apply'] = _parse_on_off(line.split()[-1])
        elif 'SEDC with' in line:
            parameters['sedc_scheme'] = {
                'OBS correction scheme': 'OBS',
                'G06 correction scheme': 'G06',
                'D3 correction scheme': 'D3',
                'D3(BJ) correction scheme': 'D3-BJ',
                'D4 correction scheme': 'D4',
                'JCHS correction scheme': 'JCHS',
                'TS correction scheme': 'TS',
                'TSsurf correction scheme': 'TSSURF',
                'TS+SCS correction scheme': 'TSSCS',
                'aperiodic TS+SCS correction scheme': 'ATSSCS',
                'aperiodic MBD@SCS method': 'AMBD',
                'MBD@SCS method': 'MBD',
                'aperiodic MBD@rsSCS method': 'AMBD*',
                'MBD@rsSCS method': 'MBD*',
                'XDM correction scheme': 'XDM',
            }[line.split(':')[-1].strip()]

        # Basis Set Parameters

        elif 'basis set accuracy' in line:
            parameters['basis_precision'] = line.split()[-1]
        elif 'plane wave basis set cut-off' in line:
            parameters['cut_off_energy'] = float(line.split()[-2])
        elif re.match(r'\sfinite basis set correction\s*:', line):
            parameters['finite_basis_corr'] = {
                'none': 0,
                'manual': 1,
                'automatic': 2,
            }[line.split()[-1]]

        # Electronic Parameters

        elif 'treating system as spin-polarized' in line:
            parameters['spin_polarized'] = True

        # Electronic Minimization Parameters

        elif 'Treating system as non-metallic' in line:
            parameters['fix_occupancy'] = True
        elif 'total energy / atom convergence tol.' in line:
            parameters['elec_energy_tol'] = float(line.split()[-2])
        elif 'convergence tolerance window' in line:
            parameters['elec_convergence_win'] = int(line.split()[-2])
        elif 'max. number of SCF cycles:' in line:
            parameters['max_scf_cycles'] = float(line.split()[-1])
        elif 'dump wavefunctions every' in line:
            parameters['num_dump_cycles'] = float(line.split()[-3])

        # Density Mixing Parameters

        elif 'density-mixing scheme' in line:
            parameters['mixing_scheme'] = line.split()[-1]

    return parameters


def _read_unit_cell(out: io.TextIOBase):
    """Read a Unit Cell block from a .castep file."""
    for line in out:
        fields = line.split()
        if len(fields) == 6:
            break
    lattice_real = []
    for _ in range(3):
        lattice_real.append([float(f) for f in fields[0:3]])
        line = out.readline()
        fields = line.split()
    return np.array(lattice_real)


def _read_forces(out: io.TextIOBase, n_atoms: int):
    """Read a block for atomic forces from a .castep file."""
    constraints: List[FixConstraint] = []
    forces = []
    for line in out:
        fields = line.split()
        if len(fields) == 7:
            break
    for n in range(n_atoms):
        consd = np.array([0, 0, 0])
        fxyz = [0.0, 0.0, 0.0]
        for i, force_component in enumerate(fields[-4:-1]):
            if force_component.count("(cons'd)") > 0:
                consd[i] = 1
            # remove constraint labels in force components
            fxyz[i] = float(force_component.replace("(cons'd)", ''))
        if consd.all():
            constraints.append(FixAtoms(n))
        elif consd.any():
            constraints.append(FixCartesian(n, consd))
        forces.append(fxyz)
        line = out.readline()
        fields = line.split()
    return forces, constraints


def _read_fractional_coordinates(out: io.TextIOBase, n_atoms: int):
    """Read fractional coordinates from a .castep file."""
    species: List[str] = []
    custom_species: Optional[List[str]] = None  # A CASTEP special thing
    positions_frac: List[List[float]] = []
    for line in out:
        fields = line.split()
        if len(fields) == 7:
            break
    for _ in range(n_atoms):
        spec_custom = fields[1].split(':', 1)
        elem = spec_custom[0]
        if len(spec_custom) > 1 and custom_species is None:
            # Add it to the custom info!
            custom_species = list(species)
        species.append(elem)
        if custom_species is not None:
            custom_species.append(fields[1])
        positions_frac.append([float(s) for s in fields[3:6]])
        line = out.readline()
        fields = line.split()
    return species, custom_species, positions_frac


def _read_stress(out: io.TextIOBase):
    """Read a block for the stress tensor from a .castep file."""
    for line in out:
        fields = line.split()
        if len(fields) == 6:
            break
    results = {}
    stress = []
    for _ in range(3):
        stress.append([float(s) for s in fields[2:5]])
        line = out.readline()
        fields = line.split()
    # stress in .castep file is given in GPa
    results['stress'] = np.array(stress) * units.GPa
    results['stress'] = results['stress'].reshape(9)[[0, 4, 8, 5, 2, 1]]
    line = out.readline()
    if "Pressure:" in line:
        results['pressure'] = float(
            line.split()[-2]) * units.GPa  # type: ignore[assignment]
    return results


def _add_atoms(
    images,
    lattice_real,
    species,
    custom_species,
    positions_frac,
    constraints,
    results,
):
    # If all the lattice parameters are fixed,
    # the `Unit Cell` block in the .castep file is not printed
    # in every ionic step.
    # The lattice paramters are therefore taken from the last step.
    # This happens:
    # - `GeometryOptimization`: `FIX_ALL_CELL : TRUE`
    # - `MolecularDynamics`: `MD_ENSEMBLE : NVE or NVT`
    if lattice_real is None:
        lattice_real = images[-1].cell.copy()

    # for highly symmetric systems (where essentially only the
    # stress is optimized, but the atomic positions) positions
    # are only printed once.
    if species is None:
        species = images[-1].symbols
    if positions_frac is None:
        positions_frac = images[-1].get_scaled_positions()

    _set_energy_and_free_energy(results)

    atoms = Atoms(
        species,
        cell=lattice_real,
        constraint=constraints,
        pbc=True,
        scaled_positions=positions_frac,
    )
    if custom_species is not None:
        atoms.new_array(
            'castep_custom_species',
            np.array(custom_species),
        )

    atoms.calc = SinglePointCalculator(atoms)
    atoms.calc.results = results

    images.append(atoms)


def _read_mulliken_charges(out: io.TextIOBase):
    """Read a block for Mulliken charges from a .castep file."""
    for i in range(3):
        line = out.readline()
        if i == 1:
            spin_polarized = 'Spin' in line
    results = defaultdict(list)
    for line in out:
        fields = line.split()
        if len(fields) == 1:
            break
        if spin_polarized:
            if len(fields) != 7:  # due to CASTEP 18 outformat changes
                results['charges'].append(float(fields[-2]))
                results['magmoms'].append(float(fields[-1]))
        else:
            results['charges'].append(float(fields[-1]))
    return {k: np.array(v) for k, v in results.items()}


def _read_hirshfeld_details(out: io.TextIOBase, n_atoms: int):
    """Read the Hirshfeld analysis when iprint > 1 from a .castep file."""
    results = defaultdict(list)
    for _ in range(n_atoms):
        for line in out:
            if line.strip() == '':
                break  # end for each atom
            if 'Hirshfeld / free atomic volume :' in line:
                line = out.readline()
                fields = line.split()
                results['hirshfeld_volume_ratios'].append(float(fields[0]))
    return {k: np.array(v) for k, v in results.items()}


def _read_hirshfeld_charges(out: io.TextIOBase):
    """Read a block for Hirshfeld charges from a .castep file."""
    for i in range(3):
        line = out.readline()
        if i == 1:
            spin_polarized = 'Spin' in line
    results = defaultdict(list)
    for line in out:
        fields = line.split()
        if len(fields) == 1:
            break
        if spin_polarized:
            results['hirshfeld_charges'].append(float(fields[-2]))
            results['hirshfeld_magmoms'].append(float(fields[-1]))
        else:
            results['hirshfeld_charges'].append(float(fields[-1]))
    return {k: np.array(v) for k, v in results.items()}


def _set_energy_and_free_energy(results: Dict[str, Any]):
    """Set values referred to as `energy` and `free_energy`."""
    if 'energy_with_dispersion_correction' in results:
        suffix = '_with_dispersion_correction'
    else:
        suffix = '_without_dispersion_correction'

    if 'free_energy' + suffix in results:  # metallic
        keye = 'energy_zero' + suffix
        keyf = 'free_energy' + suffix
    else:  # non-metallic
        # The finite basis set correction is applied to the energy at finite T
        # (not the energy at 0 K). We should hence refer to the corrected value
        # as `energy` only when the free energy is unavailable, i.e., only when
        # FIX_OCCUPANCY : TRUE and thus no smearing is applied.
        if 'energy_with_finite_basis_set_correction' in results:
            keye = 'energy_with_finite_basis_set_correction'
        else:
            keye = 'energy' + suffix
        keyf = 'energy' + suffix

    results['energy'] = results[keye]
    results['free_energy'] = results[keyf]
