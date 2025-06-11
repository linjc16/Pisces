from __future__ import annotations

import numpy as np
import pytest

import ase.build
import ase.units
from ase import Atoms
from ase.md.nose_hoover_chain import (
    IsotropicMTKBarostat,
    IsotropicMTKNPT,
    NoseHooverChainNVT,
    NoseHooverChainThermostat,
)
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


@pytest.fixture
def hcp_Cu() -> Atoms:
    atoms = ase.build.bulk(
        "Cu", crystalstructure='hcp', a=2.53, c=4.11
    ).repeat(2)
    return atoms


@pytest.mark.parametrize("tchain", [1, 3])
def test_thermostat(hcp_Cu: Atoms, tchain: int):
    atoms = hcp_Cu.copy()

    timestep = 1.0 * ase.units.fs
    thermostat = NoseHooverChainThermostat(
        num_atoms_global=len(atoms),
        masses=atoms.get_masses()[:, None],
        temperature_K=1000,
        tdamp=100 * timestep,
        tchain=tchain,
    )

    rng = np.random.default_rng(0)
    p = rng.standard_normal(size=(len(atoms), 3))

    # Forward `n` steps with `timestep` and backward `2 * n` steps with
    # `0.5 * timestep`, which should go back to the initial state.
    n = 1000
    p_start = p.copy()
    eta_start = thermostat._eta.copy()
    p_eta_start = thermostat._p_eta.copy()
    for _ in range(n):
        p = thermostat.integrate_nhc(p, timestep)
    for _ in range(2 * n):
        p = thermostat.integrate_nhc(p, -0.5 * timestep)

    assert np.allclose(p, p_start, atol=1e-6)
    assert np.allclose(thermostat._eta, eta_start, atol=1e-6)
    assert np.allclose(thermostat._p_eta, p_eta_start, atol=1e-6)


@pytest.mark.parametrize("pchain", [1, 3])
def test_isotropic_barostat(asap3, hcp_Cu: Atoms, pchain: int):
    atoms = hcp_Cu.copy()
    atoms.calc = asap3.EMT()

    timestep = 1.0 * ase.units.fs
    barostat = IsotropicMTKBarostat(
        num_atoms_global=len(atoms),
        masses=atoms.get_masses()[:, None],
        temperature_K=1000,
        pdamp=1000 * timestep,
        pchain=pchain,
    )

    rng = np.random.default_rng(0)
    p_eps = float(rng.standard_normal())

    # Forward `n` steps with `timestep` and backward `2 * n` steps with
    # `0.5 * timestep`, which should go back to the initial state.
    n = 1000
    p_eps_start = p_eps
    xi_start = barostat._xi.copy()
    p_xi_start = barostat._p_xi.copy()
    for _ in range(n):
        p_eps = barostat.integrate_nhc_baro(p_eps, timestep)
    for _ in range(2 * n):
        p_eps = barostat.integrate_nhc_baro(p_eps, -0.5 * timestep)

    assert np.allclose(p_eps, p_eps_start, atol=1e-6)
    assert np.allclose(barostat._xi, xi_start, atol=1e-6)
    assert np.allclose(barostat._p_xi, p_xi_start, atol=1e-6)


@pytest.mark.parametrize("tchain", [1, 3])
def test_nose_hoover_chain_nvt(asap3, tchain: int):
    atoms = ase.build.bulk("Cu").repeat((2, 2, 2))
    atoms.calc = asap3.EMT()

    temperature_K = 300
    rng = np.random.default_rng(0)
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=temperature_K, force_temp=True, rng=rng
    )
    Stationary(atoms)

    timestep = 1.0 * ase.units.fs
    md = NoseHooverChainNVT(
        atoms,
        timestep=timestep,
        temperature_K=temperature_K,
        tdamp=100 * timestep,
        tchain=tchain,
    )
    conserved_energy1 = md.get_conserved_energy()
    md.run(100)
    conserved_energy2 = md.get_conserved_energy()
    assert np.allclose(np.sum(atoms.get_momenta(), axis=0), 0.0)
    assert np.isclose(conserved_energy1, conserved_energy2, atol=1e-3)


@pytest.mark.parametrize("tchain", [1, 3])
@pytest.mark.parametrize("pchain", [1, 3])
def test_isotropic_mtk_npt(asap3, hcp_Cu: Atoms, tchain: int, pchain: int):
    atoms = hcp_Cu.copy()
    atoms.calc = asap3.EMT()

    temperature_K = 300
    rng = np.random.default_rng(0)
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=temperature_K, force_temp=True, rng=rng
    )
    Stationary(atoms)

    timestep = 1.0 * ase.units.fs
    md = IsotropicMTKNPT(
        atoms,
        timestep=timestep,
        temperature_K=temperature_K,
        pressure_au=10.0 * ase.units.GPa,
        tdamp=100 * timestep,
        pdamp=1000 * timestep,
        tchain=tchain,
        pchain=pchain,
    )

    conserved_energy1 = md.get_conserved_energy()
    md.run(100)
    conserved_energy2 = md.get_conserved_energy()
    assert np.allclose(np.sum(atoms.get_momenta(), axis=0), 0.0)
    assert np.isclose(conserved_energy1, conserved_energy2, atol=1e-3)
