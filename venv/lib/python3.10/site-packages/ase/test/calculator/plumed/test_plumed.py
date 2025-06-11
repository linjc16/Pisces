import numpy as np
import pytest
from pytest import approx

from ase import Atoms, units
from ase.calculators.emt import EMT
from ase.calculators.idealgas import IdealGas
from ase.calculators.lj import LennardJones
from ase.calculators.plumed import restart_from_trajectory
from ase.io.trajectory import Trajectory
from ase.md.verlet import VelocityVerlet


@pytest.mark.calculator_lite()
@pytest.mark.calculator('plumed')
def test_units(factory):
    """
    Note: if this test fails, plumed or ASE changed some units.
    It has to be fixed in the contructor of the plumed calculator.

    In this test is considered two atoms interacting through a potential with
    the form:

    (lower wall in plumed setup)
    V = k (r - r0)^2

    The values are fixed as follow:

    time = 1 ASE time units
    k = 1 kJ/(mol*nm^2)
    r = 1 Angstrom
    r0 = 1.1 nm

    considering r in nm, V = 1 kJ/mol and the forces F(r) = 2 kJ/(mol*nm) """

    set_plumed = ["e: ENERGY",                         # check energy units
                  "d: DISTANCE ATOMS=1,2",             # check distance units
                  "LOWER_WALLS ARG=d AT=1.1 KAPPA=1",  # check forces recieved
                  "DUMPMASSCHARGE FILE=mass_charge",   # check mass and charges
                  "PRINT ARG=e,d FILE=COLVAR",
                  "FLUSH STRIDE=1"]

    # execution
    atoms = Atoms('CO', positions=[[0, 0, 0], [0, 0, 1]], charges=[0, 1])

    timestep = 1
    calc = IdealGas()
    with factory.calc(calc=calc,
                      input=set_plumed,
                      timestep=timestep,
                      atoms=atoms,
                      use_charge=True) as calc:
        ener, forces = atoms.calc.compute_bias(atoms.get_positions(), 1,
                                               atoms.get_potential_energy())
        files = calc.read_plumed_files()

    # the next values are in ase units
    ase_values = {'time': 1,
                  'energy': ener,
                  'distance': 1,
                  'masses': atoms.get_masses(),
                  'charges': atoms.get_initial_charges(),
                  'forces': forces}

    # The next values are in plumed units.
    plumed_values = {'time': files['COLVAR'][0][-1],
                     'energy': files['COLVAR'][1][-1],
                     'distance': files['COLVAR'][2][-1],
                     'masses': files['mass_charge'][1],
                     'charges': files['mass_charge'][2],
                     'forces': np.array([[0, 0, -2], [0, 0, 2]])}

    assert ase_values['time'] * 1 / (1000 * units.fs) == \
        approx(plumed_values['time'], abs=1E-5), \
        "error in time units"
    assert ase_values['energy'] * units.mol / units.kJ == \
        approx(plumed_values['energy'], abs=1E-5), \
        "error in energy units"
    assert ase_values['distance'] * 1 / units.nm == \
        approx(plumed_values['distance'], abs=1E-5), \
        "error in distance units"
    assert ase_values['forces'] * units.nm * units.mol / units.kJ == \
        approx(plumed_values['forces'], abs=1E-5), \
        "error in forces units"
    assert ase_values['masses'] == approx(plumed_values['masses'],
                                          abs=1E-5), \
        "error in masses units"
    assert ase_values['charges'] == approx(plumed_values['charges'],
                                           abs=1E-5), \
        "error in charges units"


@pytest.mark.calculator_lite()
@pytest.mark.calculator('plumed')
def test_CVs(factory):
    """ This test calls plumed-ASE calculator for computing some CVs.
    Moreover, it computes those CVs directly from atoms.positions and
    compares them"""
    # plumed setting
    ps = 1000 * units.fs
    set_plumed = [f"UNITS LENGTH=A TIME={1 / ps} ENERGY={units.mol / units.kJ}",
                  "c1: COM ATOMS=1,2",
                  "c2: CENTER ATOMS=1,2",
                  "l: DISTANCE ATOMS=c1,c2",
                  "d: DISTANCE ATOMS=1,2",
                  "c: COORDINATION GROUPA=1 GROUPB=2 R_0=100 MM=0 NN=10",
                  "FLUSH STRIDE=1",
                  "PRINT ARG=d,c,l STRIDE=10 FILE=COLVAR_test1"]

    # execution
    atoms = Atoms('CO', positions=[[0, 0, 0], [0, 0, 5]])  # CO molecule
    _, colvar = run(factory, [set_plumed, atoms, 5], calc=EMT(), steps=101)

    # this compares the time calculated by ASE and plumed
    timeASE = np.arange(0., 501., 50)
    timePlumed = colvar['COLVAR_test1'][0]
    assert timeASE == approx(timePlumed), "Error in time registered by plumed"

    # This compares the distance of atoms calculated by ASE and plumed
    distASE = np.array([5., 51.338332, 141.252854, 231.167376, 321.081899,
                        410.996421, 500.910943, 590.825465, 680.739987,
                        770.654509, 860.569031])
    distPlumed = colvar['COLVAR_test1'][1]
    assert distPlumed == approx(distASE), "Error in distance"

    # this compares the coordination number calculated by ASE and plumed
    CASE = np.array([1.0000e+00, 9.9873e-01, 3.0655e-02, 2.2900e-04,
                     9.0000e-06, 1.0000e-06, 0.0000e+00, 0.0000e+00,
                     0.0000e+00, 0.0000e+00, 0.0000e+00])
    CPlumed = colvar['COLVAR_test1'][2]
    assert CASE == approx(CPlumed, abs=1E-5), "Error in coordination number"

    # this compares the distance between center of mass and geometrical center
    # calculated by ASE and plumed
    centersASE = np.array([0.355944, 3.654717, 10.05563, 16.456542, 22.857455,
                           29.258367, 35.65928, 42.060192, 48.461104,
                           54.862017, 61.262929])

    centersPlumed = colvar['COLVAR_test1'][3]
    assert centersASE == approx(centersPlumed)


@pytest.mark.calculator_lite()
@pytest.mark.calculator('plumed')
def test_metadyn(factory):
    """This test computes a Metadynamics calculation,
    This result is compared with the same calulation made externally"""
    params = setups()
    atoms, _ = run(factory, params, steps=58)

    position1 = -0.0491871
    position2 = 6.73693
    forceWithBias = 0.28807

    assert (atoms.get_positions()[0][0] == approx(position1, abs=0.01) and
            atoms.get_positions()[1][0] == approx(position2, abs=0.01)), \
        "Error in the metadynamics simulation"
    assert atoms.get_forces()[0][0] == approx(forceWithBias, abs=0.01), \
        "Error in the computation of Bias-forces"


@pytest.mark.calculator_lite()
@pytest.mark.calculator('plumed')
def test_restart(factory):
    ins = setups()
    # first steps
    _, _res = run(factory, ins, name='restart')

    # rest of steps with restart
    input, atoms1, timestep = setups()
    with restart_from_trajectory('test-restart.traj',
                                 calc=LennardJones(epsilon=10, sigma=6),
                                 input=input,
                                 timestep=timestep,
                                 atoms=atoms1) as atoms1.calc:
        with VelocityVerlet(atoms1, timestep) as dyn:
            dyn.run(30)

    # Values computed externally
    position1 = -0.0491871
    position2 = 6.73693
    forceWithBias = 0.28807

    assert atoms1.get_forces()[0][0] == approx(forceWithBias, abs=0.01), \
        "Error in restart for the computation of Bias-forces"

    assert (atoms1.get_positions()[0][0] == approx(position1, abs=0.01) and
            atoms1.get_positions()[1][0] == approx(position2, abs=0.01)), \
        "Error in the restart of metadynamics simulation"


@pytest.mark.calculator_lite()
@pytest.mark.calculator('plumed')
def test_postpro(factory):
    # Metadynamics simulation
    params = setups('direct')
    _, direct = run(factory, params, name='direct', steps=58)

    params = setups('postpro')
    # Postpro resconstruction
    with factory.calc(calc=IdealGas(),
                      input=params[0],
                      atoms=params[1],
                      timestep=params[2]) as calc:
        with Trajectory('test-direct.traj') as traj:
            postpr = calc.write_plumed_files(traj)['HILLS_postpro']

    assert postpr == approx(direct['HILLS_direct'])


@pytest.mark.calculator_lite()
@pytest.mark.calculator('plumed')
def test_pbc(factory):
    atoms = Atoms('H2')
    atoms.set_positions([[1, 0, 0], [11, 2, 0]])

    atoms.set_cell([[10, 0, 0], [10, 10, 0], [0, 0, 10]])
    traj = [atoms]

    ps = 1000 * units.fs
    setup = [f"UNITS LENGTH=A TIME={1 / ps} ENERGY={units.mol / units.kJ}",
             "d: DISTANCE ATOMS=1,2",
             "PRINT ARG=d STRIDE=100 FILE=COLVAR_pbc"]

    with factory.calc(calc=IdealGas(),
                      input=setup,
                      atoms=atoms,
                      timestep=1) as calc:
        dist = calc.write_plumed_files(traj)['COLVAR_pbc']

    assert dist[1] == 2., "Error in PBC"


def run(factory, inputs, name='',
        calc=LennardJones(epsilon=10, sigma=6),
        traj=None, steps=29):
    input, atoms, timestep = inputs
    with factory.calc(calc=calc,
                      input=input,
                      timestep=timestep,
                      atoms=atoms) as atoms.calc:
        with VelocityVerlet(atoms, timestep,
                            trajectory=f'test-{name}.traj') as dyn:
            dyn.run(steps)
        res = atoms.calc.read_plumed_files()
    return atoms, res


def setups(name=''):
    ps = 1000 * units.fs
    set_plumed = [f"UNITS LENGTH=A TIME={1 / ps} ENERGY={units.mol / units.kJ}",
                  "d: DISTANCE ATOMS=1,2",
                  "FLUSH STRIDE=1",
                  f"METAD ARG=d SIGMA=0.5 HEIGHT=2 PACE=20 FILE=HILLS_{name}"]
    atoms = Atoms('CO', positions=[[0, 0, 0], [6.7, 0, 0]])
    timestep = 0.05
    return set_plumed, atoms, timestep
