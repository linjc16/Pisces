import numpy as np
import pytest

from ase import Atoms


@pytest.fixture()
def lj_epsilons():
    # Set up original LJ epsilon energy parameter and another, modified,
    # epsilon value
    return {"eps_orig": 2.5, "eps_modified": 4.25}


@pytest.mark.calculator_lite()
@pytest.mark.calculator("lammpslib")
def test_change_cell_dimensions_and_pbc(factory, dimer_params, lj_epsilons):
    """Ensure that post_change_box commands are actually executed after
    changing the dimensions of the cell or its periodicity.  This is done by
    setting up an isolated dimer with a Lennard-Jones potential and a set of
    post_changebox_cmds that specify the same potential but with a rescaled
    energy (epsilon) parameter.  The energy is then computed twice, once before
    changing the cell dimensions and once after, and the values are compared to
    the expected values based on the two different epsilons to ensure that the
    modified LJ potential is used for the second calculation.  The procedure is
    repeated but where the periodicity of the cell boundaries is changed rather
    than the cell dimensions.
    """
    # Make a dimer in a large nonperiodic box
    dimer = Atoms(**dimer_params)
    spec, a = extract_dimer_species_and_separation(dimer)

    # Ensure LJ cutoff is large enough to encompass the dimer
    lj_cutoff = 3 * a

    calc_params = calc_params_lj_changebox(spec, lj_cutoff, **lj_epsilons)

    dimer.calc = factory.calc(**calc_params)

    energy_orig = dimer.get_potential_energy()

    # Shrink the box slightly to invalidate cached energy and force change_box
    # to be issued.  This shouldn't actually affect the energy in and of itself
    # since our dimer has non-periodic boundaries.
    cell_orig = dimer.get_cell()
    dimer.set_cell(cell_orig * 1.01, scale_atoms=False)

    energy_modified = dimer.get_potential_energy()

    eps_scaling_factor = lj_epsilons["eps_modified"] / lj_epsilons["eps_orig"]
    assert energy_modified == pytest.approx(
        eps_scaling_factor * energy_orig, rel=1e-4)

    # Reset dimer cell.  Also, create and attach new calculator so that
    # previous post_changebox_cmds won't be in effect.
    dimer.set_cell(cell_orig, scale_atoms=False)
    dimer.calc = factory.calc(**calc_params)

    # Compute energy of original configuration again so that a change_box will
    # be triggered on the next calculation after we change the pbcs
    energy_orig = dimer.get_potential_energy()

    # Change the periodicity of the cell along one direction.  This shouldn't
    # actually affect the energy in and of itself since the cell is large
    # relative to the dimer
    dimer.set_pbc([False, True, False])

    energy_modified = dimer.get_potential_energy()

    assert energy_modified == pytest.approx(
        eps_scaling_factor * energy_orig, rel=1e-4)


def calc_params_lj_changebox(spec, lj_cutoff, eps_orig, eps_modified):
    def lj_pair_style_coeff_lines(lj_cutoff, eps):
        return [f"pair_style lj/cut {lj_cutoff}", f"pair_coeff * * {eps} 1"]

    # Set up LJ pair style using original epsilon and define a modified LJ to
    # be executed after change_box using the modified epsilon
    calc_params = {}
    calc_params["lmpcmds"] = lj_pair_style_coeff_lines(lj_cutoff, eps_orig)
    calc_params["atom_types"] = {spec: 1}
    calc_params["log_file"] = "test.log"
    calc_params["keep_alive"] = True
    calc_params["post_changebox_cmds"] = lj_pair_style_coeff_lines(
        lj_cutoff, eps_modified
    )
    return calc_params


def extract_dimer_species_and_separation(atoms):
    """
    Given a monoatomic dimer, extract the species of its atoms and their
    separation
    """
    # Extract species
    if len(set(atoms.symbols)) > 1:
        raise ValueError("Dimer must contain only one atomic species")
    spec = atoms.symbols[0]

    # Get dimer separation
    pos = atoms.get_positions()
    a = np.linalg.norm(pos[1] - pos[0])
    return spec, a
