import pytest


@pytest.fixture()
def calc_params_NiH():
    calc_params = {}
    calc_params["lmpcmds"] = [
        "pair_style eam/alloy",
        "pair_coeff * * NiAlH_jea.eam.alloy Ni H",
    ]
    calc_params["atom_types"] = {"Ni": 1, "H": 2}
    calc_params["log_file"] = "test.log"
    calc_params["keep_alive"] = True
    return calc_params


@pytest.fixture()
def dimer_params():
    dimer_params = {}
    a = 2.0
    dimer_params["symbols"] = "Ni" * 2
    dimer_params["positions"] = [(0, 0, 0), (a, 0, 0)]
    dimer_params["cell"] = (1000 * a, 1000 * a, 1000 * a)
    dimer_params["pbc"] = (False, False, False)
    return dimer_params
