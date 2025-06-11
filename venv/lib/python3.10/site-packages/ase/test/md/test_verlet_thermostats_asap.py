import numpy as np
import pytest

from ase.build import bulk
from ase.io import Trajectory, read
from ase.md import Andersen, Langevin, VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.units import fs


# test Verlet, Langevin and Andersen with asap3
@pytest.mark.slow()
def test_verlet_thermostats_asap(asap3, testdir, allraise):
    rng = np.random.RandomState(0)
    calculator = asap3.EMT()
    T_low = 10
    T_high = 300
    md_kwargs = {'timestep': 0.5 * fs, 'logfile': '-', 'loginterval': 500}

    a = bulk('Au').repeat((4, 4, 4))
    a[5].symbol = 'Ag'

    # test thermalization by MaxwellBoltzmannDistribution
    thermalize(T_high, a, rng)
    assert abs(a.get_temperature() - T_high) < 0.0001

    # test conservation of total energy e0 using Verlet
    a_verlet, traj = prepare_md(a, calculator)
    with traj:
        e0 = a_verlet.get_total_energy()
        with VelocityVerlet(a_verlet, **md_kwargs) as md:
            md.attach(traj.write, 100)
            md.run(steps=10000)
        traj_verlet = read('Au7Ag.traj', index=':')
    assert abs(traj_verlet[-1].get_total_energy() - e0) < 0.0001

    # test reproduction of Verlet by Langevin and Andersen for thermostats
    # switched off
    pos_verlet = [t.get_positions() for t in traj_verlet[:3]]
    md_kwargs.update({'temperature_K': T_high})
    for MDalgo in [Langevin, Andersen]:
        a_md, traj = prepare_md(a, calculator)
        with traj:
            kw = dict(md_kwargs)
            kw.update(rng=rng)
            if MDalgo is Langevin:
                kw['friction'] = 0.0
            elif MDalgo is Andersen:
                kw['andersen_prob'] = 0.0

            with MDalgo(a_md, **kw) as md:
                md.attach(traj, 100)
                md.run(steps=200)
                traj_md = read('Au7Ag.traj', index=':')
                pos_md = [t.get_positions() for t in traj_md[:3]]
                assert np.allclose(pos_verlet, pos_md)  # Verlet reproduced?

                # test thermalization to target temperature by thermostats and
                # conservation of average temperature by thermostats
                md.set_timestep(4 * fs)
                if MDalgo is Langevin:
                    md.set_friction(0.01)
                elif MDalgo is Andersen:
                    md.set_andersen_prob(0.01)
                # thermalize with low temperature (T)
                thermalize(T_low, a_md, rng)
                assert abs(a_md.get_temperature() - T_low) < 0.0001

                # equilibration, i.e. thermalization to high T
                md.run(steps=500)
                temp = []

                def recorder():
                    temp.append(a_md.get_temperature())
                md.attach(recorder, interval=1)
                md.run(7000)
                temp = np.array(temp)
                avgtemp = np.mean(temp)
                fluct = np.std(temp)
                print("Temperature is {:.2f} K +/- {:.2f} K".format(avgtemp,
                                                                    fluct))
                assert abs(avgtemp - T_high) < 10.0


def prepare_md(atoms, calculator):
    a_md = atoms.copy()
    a_md.calc = calculator
    traj = Trajectory('Au7Ag.traj', 'w', a_md)
    return a_md, traj


def thermalize(temp, atoms, rng):
    MaxwellBoltzmannDistribution(atoms, temperature_K=temp, force_temp=True,
                                 rng=rng)
    Stationary(atoms)
