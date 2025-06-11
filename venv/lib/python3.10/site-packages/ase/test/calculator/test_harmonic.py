import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.integrate import trapezoid

from ase import Atoms
from ase.calculators.calculator import CalculationFailed, CalculatorSetupError
from ase.calculators.emt import EMT
from ase.calculators.harmonic import HarmonicCalculator, HarmonicForceField
from ase.calculators.mixing import MixedCalculator
from ase.geometry.geometry import get_distances_derivatives
from ase.md.andersen import Andersen
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import BFGS
from ase.units import fs
from ase.vibrations import Vibrations

ref_pos = np.asarray([[8.7161, 7.96276, 8.48206], [8.60594, 8.04985, 9.44464],
                      [8.0154, 8.52264, 8.10545]])
ref_atoms = Atoms('OH2', positions=ref_pos)  # relaxed water molecule
ref_energy = -14.222189  # y shift of the 'parabola' (harmonic potential)

# example Hessian matrix as obtained from DFT
hessian_x = np.asarray([[2.82630333e+01, -2.24763667e+01, 7.22478333e+00,
                         -2.96970000e+00, 2.34363333e+00, 2.72788333e+00,
                         -2.52159833e+01, 2.01307833e+01, -9.94651667e+00],
                        [-2.24763667e+01, 1.78621333e+01, -5.77378333e+00,
                         2.33703333e+00, -1.85276667e+00, -2.15118333e+00,
                         2.01258667e+01, -1.60350833e+01, 7.93248333e+00],
                        [7.22478333e+00, -5.77378333e+00, 5.72735000e+01,
                         7.25470000e+00, -5.75313333e+00, -4.69477333e+01,
                         -1.44613000e+01, 1.15504833e+01, -1.03523333e+01],
                        [-2.96970000e+00, 2.33703333e+00, 7.25470000e+00,
                         2.96170000e+00, -2.36901667e+00, -3.76841667e+00,
                         -2.83833333e-02, 3.06833333e-02, -3.49190000e+00],
                        [2.34363333e+00, -1.85276667e+00, -5.75313333e+00,
                         -2.36901667e+00, 1.89046667e+00, 2.95495000e+00,
                         2.90666667e-02, -1.80666667e-02, 2.79565000e+00],
                        [2.72788333e+00, -2.15118333e+00, -4.69477333e+01,
                         -3.76841667e+00, 2.95495000e+00, 4.89340000e+01,
                         1.03146667e+00, -8.18450000e-01, -1.96118333e+00],
                        [-2.52159833e+01, 2.01258667e+01, -1.44613000e+01,
                         -2.83833333e-02, 2.90666667e-02, 1.03146667e+00,
                         2.52034000e+01, -2.01516833e+01, 1.34293167e+01],
                        [2.01307833e+01, -1.60350833e+01, 1.15504833e+01,
                         3.06833333e-02, -1.80666667e-02, -8.18450000e-01,
                         -2.01516833e+01, 1.60592333e+01, -1.07369667e+01],
                        [-9.94651667e+00, 7.93248333e+00, -1.03523333e+01,
                         -3.49190000e+00, 2.79565000e+00, -1.96118333e+00,
                         1.34293167e+01, -1.07369667e+01, 1.23150000e+01]])


def assert_water_is_relaxed(atoms):
    forces = atoms.get_forces()
    assert np.allclose(np.zeros(forces.shape), forces)
    assert np.allclose(ref_energy, atoms.get_potential_energy())
    assert np.allclose(atoms.get_angle(1, 0, 2), ref_atoms.get_angle(1, 0, 2))
    assert np.allclose(atoms.get_distance(0, 1), ref_atoms.get_distance(0, 1))
    assert np.allclose(atoms.get_distance(0, 2), ref_atoms.get_distance(0, 2))


def run_optimize(atoms):
    opt = BFGS(atoms)
    opt.run(fmax=1e-9)


@pytest.mark.optimize()
def test_cartesians():
    """In Cartesian coordinates the first 6 trash eigenvalues (translations and
    rotations) can be slightly different from zero; hence set them to zero
    using an increased parameter zero_thresh.
    """
    zero_thresh = 0.06  # set eigvals to zero if abs(eigenvalue) < zero_thresh
    hff = HarmonicForceField(ref_atoms=ref_atoms, ref_energy=ref_energy,
                             hessian_x=hessian_x, zero_thresh=zero_thresh)
    assert np.allclose(hff.hessian_q, hff.hessian_x)
    atoms = ref_atoms.copy()
    atoms.calc = HarmonicCalculator(hff)
    assert_water_is_relaxed(atoms)  # atoms has not been distorted
    run_optimize(atoms)             # nothing should happen
    assert_water_is_relaxed(atoms)  # atoms should still be relaxed
    atoms.set_distance(0, 1, 3.5)   # now distort atoms along axis, no rotation
    run_optimize(atoms)             # optimization should recover original
    assert_water_is_relaxed(atoms)    # relaxed geometry

    with pytest.raises(AssertionError):
        atoms.rattle()                  # relaxation should fail to recover the
        atoms.rotate(90, 'x')           # original geometry of the atoms,
        run_optimize(atoms)             # because Cartesian coordinates are
        assert_water_is_relaxed(atoms)  # not rotationally invariant.


@pytest.mark.optimize()
def test_constraints_with_cartesians():
    """Project out forces along x-component of H-atom (index 0 in the q-vector
    with the Cartesian coordinates (here: x=q)). A change in the x-component of
    the H-atom should not result in restoring forces, when they were projected
    out from the Hessian matrix.
    """
    def test_forces(calc):
        atoms = ref_atoms.copy()
        atoms.calc = calc
        pos = ref_pos.copy()
        pos[0, 0] *= 2
        atoms.set_positions(pos)
        run_optimize(atoms)  # (no) restoring force along distorted x-component
        xdiff = atoms.get_positions() - ref_pos
        return all(xdiff[xdiff != 0] == pos[0, 0] / 2)

    zero_thresh = 0.06  # set eigvals to zero if abs(eigenvalue) < zero_thresh
    parameters = {'ref_atoms': ref_atoms, 'ref_energy': ref_energy,
                  'hessian_x': hessian_x, 'zero_thresh': zero_thresh}
    hff = HarmonicForceField(**parameters)
    calc = HarmonicCalculator(hff)
    assert not test_forces(calc)  # restoring force along distorted x-component

    parameters['constrained_q'] = [0]  # project out the coordinate with index 0
    hff = HarmonicForceField(**parameters)
    calc = HarmonicCalculator(hff)
    assert test_forces(calc)  # no restoring force along distorted x-component


def setup_water(calc):
    atoms = ref_atoms.copy()
    atoms.calc = calc
    assert_water_is_relaxed(atoms)
    atoms.rattle(0.3)
    atoms.rotate(160, 'x')
    assert not np.allclose(ref_energy, atoms.get_potential_energy())
    return atoms


# start doc example 3
dist_defs = [[0, 1], [1, 2], [2, 0]]  # define three distances by atom indices


def water_get_q_from_x(atoms):
    """Simple internal coordinates to describe water with three distances."""
    q_vec = [atoms.get_distance(i, j) for i, j in dist_defs]
    return np.asarray(q_vec)


def water_get_jacobian(atoms):
    """Function to return the Jacobian for the water molecule described by
    three distances."""
    pos = atoms.get_positions()
    dist_vecs = [pos[j] - pos[i] for i, j in dist_defs]
    derivs = get_distances_derivatives(dist_vecs)
    jac = []
    for i, defin in enumerate(dist_defs):
        dqi_dxj = np.zeros(ref_pos.shape)
        for j, deriv in enumerate(derivs[i]):
            dqi_dxj[defin[j]] = deriv
        jac.append(dqi_dxj.flatten())
    return np.asarray(jac)
# end doc example 3


def test_raise_Errors():
    with pytest.raises(CalculatorSetupError):
        HarmonicForceField(ref_atoms=ref_atoms, hessian_x=hessian_x,
                           get_q_from_x=lambda x: x)
    with pytest.raises(CalculatorSetupError):
        HarmonicForceField(ref_atoms=ref_atoms, hessian_x=hessian_x,
                           variable_orientation=True)
    with pytest.raises(CalculatorSetupError):
        HarmonicForceField(ref_atoms=ref_atoms, hessian_x=hessian_x,
                           cartesian=False)
    with pytest.raises(CalculationFailed):
        hff = HarmonicForceField(ref_atoms=ref_atoms, ref_energy=ref_energy,
                                 hessian_x=hessian_x,
                                 get_q_from_x=water_get_q_from_x,
                                 get_jacobian=lambda x: np.ones((3, 9)),
                                 cartesian=True, variable_orientation=True)
        calc = HarmonicCalculator(hff)
        setup_water(calc)


@pytest.mark.optimize()
def test_internals():
    parameters = {'ref_atoms': ref_atoms, 'ref_energy': ref_energy,
                  'hessian_x': hessian_x, 'get_q_from_x': water_get_q_from_x,
                  'get_jacobian': water_get_jacobian, 'cartesian': False}
    hff = HarmonicForceField(**parameters)  # calculation in internals
    calc = HarmonicCalculator(hff)
    atoms = setup_water(calc)  # distorted copy of ref_atoms
    run_optimize(atoms)        # recover original configuration
    assert_water_is_relaxed(atoms)

    parameters['cartesian'] = True  # calculation in Cartesian Coordinates
    hff = HarmonicForceField(**parameters)
    calc = HarmonicCalculator(hff)
    atoms = setup_water(calc)       # 'variable_orientation' not set to True!
    # but water has rotational degrees of freedom
    run_optimize(atoms)
    with pytest.raises(AssertionError):  # hence forces were incorrect
        assert_water_is_relaxed(atoms)   # original configuration not recovered

    parameters['variable_orientation'] = True
    hff = HarmonicForceField(**parameters)
    calc = HarmonicCalculator(hff)
    atoms = setup_water(calc)
    run_optimize(atoms)
    assert_water_is_relaxed(atoms)  # relaxation succeeded despite rotation


@pytest.mark.optimize()
def test_compatible_with_ase_vibrations():
    atoms = ref_atoms.copy()
    atoms.calc = EMT()
    run_optimize(atoms)
    opt_atoms = atoms.copy()
    opt_energy = atoms.get_potential_energy()
    vib = Vibrations(atoms, nfree=2)
    vib.run()
    energies = vib.get_energies()
    vib_data = vib.get_vibrations()
    hessian_2d = vib_data.get_hessian_2d()
    vib.clean()

    hff = HarmonicForceField(ref_atoms=opt_atoms, ref_energy=opt_energy,
                             hessian_x=hessian_2d)
    calc_harmonic = HarmonicCalculator(hff)
    atoms = ref_atoms.copy()
    atoms.calc = calc_harmonic
    vib = Vibrations(atoms, nfree=4, delta=1e-5)
    vib.run()
    assert np.allclose(energies, vib.get_energies())
    vib.clean()
    hff = HarmonicForceField(ref_atoms=ref_atoms, ref_energy=ref_energy,
                             hessian_x=hessian_2d,
                             get_q_from_x=water_get_q_from_x,
                             get_jacobian=water_get_jacobian, cartesian=True)
    calc_harmonic = HarmonicCalculator(hff)
    atoms = ref_atoms.copy()
    atoms.calc = calc_harmonic
    vib = Vibrations(atoms, nfree=4, delta=1e-5)
    vib.run()  # 3 transl and 3 rot are removed by internal coordinates
    assert_array_almost_equal(energies[-3:], vib.get_energies()[-3:], decimal=2)


def test_thermodynamic_integration():
    parameters = {'ref_atoms': ref_atoms, 'ref_energy': ref_energy,
                  'hessian_x': hessian_x, 'get_q_from_x': water_get_q_from_x,
                  'get_jacobian': water_get_jacobian, 'cartesian': True,
                  'variable_orientation': True}
    hff_1 = HarmonicForceField(**parameters)
    calc_harmonic_1 = HarmonicCalculator(hff_1)
    parameters['cartesian'] = False
    hff_0 = HarmonicForceField(**parameters)
    calc_harmonic_0 = HarmonicCalculator(hff_0)
    ediffs = {}  # collect energy difference for varying lambda coupling
    lambs = [0.00, 0.25, 0.50, 0.75, 1.00]  # integration grid
    for lamb in lambs:
        ediffs[lamb] = []
        calc_linearCombi = MixedCalculator(calc_harmonic_0, calc_harmonic_1,
                                           1 - lamb, lamb)
        atoms = ref_atoms.copy()
        atoms.calc = calc_linearCombi
        MaxwellBoltzmannDistribution(atoms, temperature_K=300, force_temp=True)
        Stationary(atoms)
        ZeroRotation(atoms)
        with Andersen(atoms, 0.5 * fs, temperature_K=300, andersen_prob=0.05,
                      fixcm=False) as dyn:
            for _ in dyn.irun(50):  # should be much longer for production runs
                e0, e1 = calc_linearCombi.get_energy_contributions(atoms)
                ediffs[lamb].append(float(e1) - float(e0))
            ediffs[lamb] = np.mean(ediffs[lamb])
    dA = trapezoid([ediffs[lamb] for lamb in lambs], x=lambs)  # anharm. corr.
    assert -0.005 < dA < 0.005  # the MD run is to short for convergence
    if dA == 0.0:
        raise ValueError('there is most likely something wrong, but it could '
                         'also be sheer coincidence')
