import os

import pytest

from ase.build import bulk, molecule
from ase.calculators.vasp import Vasp


@pytest.fixture()
def nacl():
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=4.1, cubic=True)
    return atoms


@pytest.fixture()
def nh3():
    atoms = molecule("NH3", vacuum=10)
    return atoms


def get_suffixes(ppp_list):
    suffixes = []
    for p in ppp_list:
        name = p.split("/")[-2]
        # since the H PPs with fractional valence
        # do not have an '_', we need to handle them
        element = name.split("_")[0] if "." not in name else "H"
        suffix = name[len(element):]
        suffixes.append(suffix)
    return suffixes


@pytest.mark.skipif(
    "VASP_PP_PATH" not in os.environ, reason="VASP_PP_PATH not set"
)
def test_potcar_setups(nacl):
    setups = {
        "recommended": ["_pv", ""],
        "GW": ["_sv_GW", "_GW"],
        "custom": ["", "_h"],
    }
    calc = Vasp(setups="recommended")
    calc.initialize(nacl)
    assert get_suffixes(calc.ppp_list) == setups["recommended"]

    calc = Vasp(setups="GW")
    calc.initialize(nacl)
    assert get_suffixes(calc.ppp_list) == setups["GW"]

    calc = Vasp(setups={"base": "minimal", "Cl": "_h"})
    calc.initialize(nacl)
    assert get_suffixes(calc.ppp_list) == setups["custom"]


@pytest.mark.skipif(
    "VASP_PP_PATH" not in os.environ, reason="VASP_PP_PATH not set"
)
def test_potcar_setups_fractional_valence(nh3):
    setups = {"base": "recommended", 1: "H.5", 2: "H1.75", 3: "H.75"}
    calc = Vasp(setups=setups, xc="PBE")
    calc.initialize(nh3)
    assert get_suffixes(calc.ppp_list) == [".5", "1.75", ".75", ""]
