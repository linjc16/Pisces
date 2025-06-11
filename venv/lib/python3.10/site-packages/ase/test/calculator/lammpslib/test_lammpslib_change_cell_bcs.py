import pytest

from ase.lattice.cubic import FaceCenteredCubic


@pytest.fixture()
def lattice_params():
    lattice_params = {}
    lattice_params["size"] = (2, 2, 2)
    lattice_params["latticeconstant"] = 3.52
    lattice_params["symbol"] = "Ni"
    lattice_params["pbc"] = True
    return lattice_params


@pytest.mark.calculator_lite()
@pytest.mark.calculator("lammpslib")
def test_lammpslib_change_cell_bcs(factory, lattice_params, calc_params_NiH):
    """Test that a change in unit cell boundary conditions is
    handled correctly by lammpslib"""

    atoms = FaceCenteredCubic(**lattice_params)

    calc = factory.calc(**calc_params_NiH)
    atoms.calc = calc

    energy_ppp_ref = -142.400000403
    energy_ppp = atoms.get_potential_energy()
    print(f"Computed energy with boundary ppp = {energy_ppp}")
    assert energy_ppp == pytest.approx(energy_ppp_ref, rel=1e-4)

    atoms.set_pbc((False, False, True))
    energy_ssp_ref = -114.524625705
    energy_ssp = atoms.get_potential_energy()
    print(f"Computed energy with boundary ssp = {energy_ssp}")
    assert energy_ssp == pytest.approx(energy_ssp_ref, rel=1e-4)
