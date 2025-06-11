import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk, make_supercell
from ase.md.bussi import Bussi
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.units import GPa, bar, fs


@pytest.fixture(scope='module')
def dynamicsparams():
    """Parameters for the Dynamics."""
    Bgold = 220.0 * GPa  # Bulk modulus of gold, in bar (1 GPa = 10000 bar)
    taut = 1000 * fs
    taup = 1000 * fs
    nvtparam = dict(temperature_K=300, taut=taut)
    nptparam = dict(temperature_K=300, pressure_au=5000 * bar, taut=taut,
                    taup=taup, compressibility_au=1 / Bgold)
    langevinparam = dict(temperature_K=300, friction=1 / (2 * taut))
    nhparam = dict(temperature_K=300, tdamp=taut)
    # NPT uses different units.  The factor 1.3 is the bulk modulus of gold in
    # ev/Å^3
    nptoldparam = dict(temperature_K=300, ttime=taut,
                       externalstress=5000 * bar,
                       pfactor=taup**2 * 1.3)
    return dict(
        nvt=nvtparam,
        npt=nptparam,
        langevin=langevinparam,
        nosehoover=nhparam,
        nptold=nptoldparam
        )


def equilibrate(atoms, dynamicsparams):
    """Make an atomic system with equilibrated temperature and pressure."""
    rng = np.random.RandomState(42)
    # Must be small enough that we can see the an off-by-one error
    # in the energy
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, force_temp=True,
                                 rng=rng)
    Stationary(atoms)
    assert abs(atoms.get_temperature() - 300) < 0.0001
    with NPTBerendsen(atoms, timestep=20 * fs, logfile='-',
                      loginterval=200,
                      **dynamicsparams['npt']) as md:
        # Equilibrate for 20 ps
        md.run(steps=1000)
    T = atoms.get_temperature()
    pres = -atoms.get_stress(
        include_ideal_gas=True)[:3].sum() / 3 / GPa * 10000
    print(f"Temperature: {T:.2f} K    Pressure: {pres:.2f} bar")
    return atoms


@pytest.fixture(scope='module')
def equilibrated(asap3, dynamicsparams):
    atoms = bulk('Au', cubic=True)
    atoms.calc = asap3.EMT()

    return equilibrate(atoms, dynamicsparams)


@pytest.fixture(scope='module')
def equilibrated_upper_tri(asap3, dynamicsparams):
    atoms = make_supercell(bulk('Pt', cubic=True),
                           [[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    atoms.calc = asap3.EMT()
    return equilibrate(atoms, dynamicsparams)


@pytest.fixture(scope='module')
def equilibrated_lower_tri(asap3, dynamicsparams):
    atoms = bulk('Pt') * (3, 1, 1)
    atoms.calc = asap3.EMT()

    # Rotate to lower triangular cell matrix
    atoms.set_cell(atoms.cell.standard_form()[0], scale_atoms=True)

    return equilibrate(atoms, dynamicsparams)


def propagate(atoms,
              asap3,
              algorithm,
              algoargs,
              max_pressure_error=None,
              com_not_thermalized=False
    ):
    print(f'Propagating algorithm in {str(algorithm)}.')
    T = []
    p = []
    with algorithm(
            atoms,
            timestep=20 * fs,
            logfile='-',
            loginterval=1000,
            **algoargs) as md:
        # Gather 2000 data points for decent statistics
        for _ in range(2000):
            md.run(5)
            T.append(atoms.get_temperature())
            pres = - atoms.get_stress(include_ideal_gas=True)[:3].sum() / 3
            p.append(pres)
    Tmean = np.mean(T)
    p = np.array(p)
    pmean = np.mean(p)
    print('Temperature: {:.2f} K +/- {:.2f} K  (N={})'.format(
        Tmean, np.std(T), len(T)))
    print('Center-of-mass corrected temperature: {:.2f} K'.format(
        Tmean * len(atoms) / (len(atoms) - 1)))
    print('Pressure: {:.2f} bar +/- {:.2f} bar  (N={})'.format(
        pmean / bar, np.std(p) / bar, len(p)))
    # Temperature error: We should be able to detect a error of 1/N_atoms
    # The factor .67 is arbitrary, smaller than 1.0 so we consistently
    # detect errors, but not so small that we get false positives.
    maxtemperr = 0.67 * 1 / len(atoms)
    targettemp = algoargs['temperature_K']
    if com_not_thermalized:
        targettemp *= (len(atoms) - 1) / len(atoms)
    assert abs(Tmean - targettemp) < maxtemperr * targettemp
    if max_pressure_error:
        try:
            # Different algorithms use different keywords
            targetpressure = algoargs['pressure_au']
        except KeyError:
            targetpressure = algoargs['externalstress']
        assert abs(pmean - targetpressure) < max_pressure_error


# Not a real optimizer test but uses optimizers.
# We should probably not mark this (in general)
@pytest.mark.optimize()
@pytest.mark.slow()
def test_nvtberendsen(asap3, equilibrated, dynamicsparams, allraise):
    propagate(Atoms(equilibrated), asap3,
              NVTBerendsen, dynamicsparams['nvt'])


@pytest.mark.optimize()
@pytest.mark.slow()
def test_langevin(asap3, equilibrated, dynamicsparams, allraise):
    propagate(Atoms(equilibrated), asap3,
              Langevin, dynamicsparams['langevin'])


@pytest.mark.optimize()
@pytest.mark.slow()
def test_bussi(asap3, equilibrated, dynamicsparams, allraise):
    propagate(Atoms(equilibrated), asap3,
              Bussi, dynamicsparams['nvt'])


@pytest.mark.optimize()
@pytest.mark.slow()
def test_nosehoovernvt(asap3, equilibrated, dynamicsparams, allraise):
    propagate(Atoms(equilibrated), asap3,
              NoseHooverChainNVT, dynamicsparams['nosehoover'])


@pytest.mark.optimize()
@pytest.mark.slow()
def test_nptberendsen(asap3, equilibrated, dynamicsparams, allraise):
    propagate(Atoms(equilibrated), asap3, NPTBerendsen,
              dynamicsparams['npt'], max_pressure_error=25.0 * bar)


@pytest.mark.optimize()
@pytest.mark.slow()
def test_npt_cubic(asap3, equilibrated, dynamicsparams, allraise):
    propagate(Atoms(equilibrated), asap3, NPT,
              dynamicsparams['nptold'],
              max_pressure_error=100 * bar,
              com_not_thermalized=True)
    # Unlike NPTBerendsen, NPT assumes that the center of mass is not
    # thermalized, so the kinetic energy should be 3/2 ' kB * (N-1) * T


@pytest.mark.optimize()
@pytest.mark.slow()
def test_npt_upper_tri(asap3, equilibrated_upper_tri, dynamicsparams, allraise):
    # Otherwise, parameters are the same as test_npt
    propagate(Atoms(equilibrated_upper_tri),
              asap3,
              NPT,
              dynamicsparams['nptold'],
              max_pressure_error=100 * bar,
              com_not_thermalized=True)


@pytest.mark.optimize()
@pytest.mark.slow()
def test_npt_lower_tri(asap3, equilibrated_lower_tri, dynamicsparams, allraise):
    # Otherwise, parameters are the same as test_npt
    propagate(Atoms(equilibrated_lower_tri),
              asap3,
              NPT,
              dynamicsparams['nptold'],
              max_pressure_error=100 * bar,
              com_not_thermalized=True)
