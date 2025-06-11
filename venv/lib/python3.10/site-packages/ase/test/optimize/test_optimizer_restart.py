import shutil
from math import sqrt
from os.path import getsize
from pathlib import Path

import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.filters import FrechetCellFilter
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, BFGSLineSearch, CellAwareBFGS


def params(opt):
    run_params = dict(fmax=0.005)
    opt_params = {}

    if opt is CellAwareBFGS:
        run_params.update(dict(smax=0.0005))

    if opt is BFGS:
        opt_params = {"append_trajectory": True}

    return run_params, opt_params


def opt_filter_atoms(opt, trajectory, restart, opt_params):
    atoms = bulk("Au")
    atoms *= 2
    atoms.rattle(stdev=0.005, seed=1)

    # if restarting, we take atoms from traj
    if Path(trajectory).is_file():
        with Trajectory(trajectory, 'r') as traj:
            atoms = traj[-1]

    atoms.calc = EMT()

    opt_relax = opt(
        FrechetCellFilter(atoms, exp_cell_factor=1.0),
        alpha=70,
        trajectory=trajectory,
        restart=restart,
        **opt_params,
    )

    return opt_relax


def fragile_optimizer(opt, trajectory, restart, run_kwargs, opt_params):
    break_now = 2

    with opt_filter_atoms(
        opt=opt, trajectory=trajectory, restart=restart, opt_params=opt_params
    ) as fragile_init:
        if isinstance(fragile_init, CellAwareBFGS):
            smax = run_kwargs.pop('smax')
            fragile_init.smax = smax
        for idx, _ in enumerate(fragile_init.irun(**run_kwargs)):
            if idx == break_now:
                break
        else:
            raise RuntimeError(
                'Fragile Optimizer did not break. Check if nsteps is to large.'
            )

    # pick up where we left off, assert we have written the files, and they
    # contain data. We check this here since these files are required in
    # order to properly restart.
    assert fragile_init.nsteps == break_now
    assert Path(restart).is_file() and Path(trajectory).is_file()
    assert all(size != 0 for size in [getsize(restart), getsize(trajectory)])

    if not opt_params.get('append_trajectory', False):
        shutil.copy(trajectory, trajectory + '.orig')

    with opt_filter_atoms(
        opt=opt, trajectory=trajectory, restart=restart, opt_params=opt_params
    ) as fragile_restart:
        if isinstance(fragile_restart, CellAwareBFGS):
            fragile_restart.smax = smax
        fragile_restart.run(**run_kwargs)

    return fragile_init, fragile_restart


@pytest.mark.parametrize(
    "opt", [BFGS, CellAwareBFGS, pytest.param(
        BFGSLineSearch, marks=pytest.mark.xfail(
            reason='Restart is absolutely broken and does not work. orig_cell '
                   'is not stored in output'))])
def test_optimizers_restart(testdir, opt):
    restart_filename = f"restart_{opt.__name__}.dat"
    trajectory_filename = f"{opt.__name__}.traj"
    run_kwargs, opt_params = params(opt)

    # single run
    with opt_filter_atoms(
        opt=opt,
        trajectory="single_" + trajectory_filename,
        restart="single_" + restart_filename,
        opt_params=opt_params,
    ) as single:
        single.run(**run_kwargs)

    # fragile restart
    fragile_init, fragile_restart = fragile_optimizer(
        opt=opt,
        trajectory="fragile_" + trajectory_filename,
        restart="fragile_" + restart_filename,
        run_kwargs=run_kwargs,
        opt_params=opt_params,
    )

    assert single.nsteps == fragile_init.nsteps + fragile_restart.nsteps

    single_traj = read_traj("single_" + trajectory_filename)
    fragile_traj = read_traj("fragile_" + trajectory_filename)

    if not opt_params.get('append_trajectory', False):
        fragile_traj_og = read_traj("fragile_" + trajectory_filename + '.orig')
        count = len(fragile_traj_og) - 1
        for traj_idx in fragile_traj:
            if traj_idx['step'] == 0:
                continue
            traj_idx['step'] += count
            fragile_traj_og.append(traj_idx)
        fragile_traj = fragile_traj_og

    # last step of init == first step of restart == single run break_now step
    # last step of restart == last step of single run
    for f_traj, s_traj in zip(fragile_traj, single_traj):
        for f, s in zip(f_traj, s_traj):
            assert np.allclose(f_traj[f], s_traj[s])


def read_traj(file: str):
    data = []

    with Trajectory(file, "r") as traj:
        for idx, atoms in enumerate(traj):

            pos = atoms.get_positions()
            forces = atoms.calc.results["forces"]
            stress = atoms.calc.results["stress"]
            energy = atoms.calc.results["energy"]

            fmax = sqrt((forces**2).sum(axis=1).max())
            smax = abs(stress).max()

            tmp = {
                "step": idx,
                "energy": energy,
                "position": pos,
                "forces": forces,
                "fmax": fmax,
                "smax": smax,
            }
            data.append(tmp)
        traj.close()

    return data
