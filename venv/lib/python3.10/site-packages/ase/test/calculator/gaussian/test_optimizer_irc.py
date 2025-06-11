import pytest

from ase import Atoms
from ase.calculators.gaussian import Gaussian, GaussianIRC, GaussianOptimizer
from ase.io import read
from ase.optimize import LBFGS


@pytest.fixture()
def atoms():
    return Atoms('CHO',
                 [[0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.35],
                  [1.178513, 0.0, -0.416662]],
                 magmoms=[0.5, 0.0, 0.5])


def get_calc(**kwargs):
    kwargs.update(mem='100MW', method='hf', basis='sto-3g')
    return Gaussian(**kwargs)


def test_optimizer(atoms, gaussian_factory):
    pos = atoms.positions.copy()
    atoms.calc = get_calc(label='opt', scf='qc')
    opt_gauss = GaussianOptimizer(atoms)
    opt_gauss.run(fmax='tight')
    e_gaussopt = read('opt.log', index=-1).get_potential_energy()

    atoms.positions[:] = pos
    atoms.calc.set_label('sp')
    with LBFGS(atoms) as opt_ase:
        opt_ase.run(fmax=1e-2)
    e_aseopt = atoms.get_potential_energy()
    assert e_gaussopt - e_aseopt == pytest.approx(0., abs=1e-3)


def test_irc(atoms, gaussian_factory):
    calc_ts = get_calc(label='ts', chk='ts.chk')
    ts = GaussianOptimizer(atoms, calc_ts)
    ts.run(opt='calcall,ts,noeigentest')
    tspos = atoms.positions.copy()

    atoms.calc = get_calc(label='sp', chk='sp.chk', freq='')
    e_ts = atoms.get_potential_energy()

    calc_irc_for = get_calc(label='irc_for', oldchk='sp', chk='irc_for.chk')
    irc_for = GaussianIRC(atoms, calc_irc_for)
    irc_for.run(direction='forward', irc='rcfc')
    e_for = read('irc_for.log', index=-1).get_potential_energy()

    atoms.positions[:] = tspos
    calc_irc_rev = get_calc(label='irc_rev', oldchk='sp', chk='irc_rev.chk')
    irc_rev = GaussianIRC(atoms, calc_irc_rev)
    irc_rev.run(direction='reverse', irc='rcfc')
    e_rev = read('irc_rev.log', index=-1).get_potential_energy()

    assert e_ts - e_for == pytest.approx(1.282, abs=1e-3)
    assert e_ts - e_rev == pytest.approx(0.201, abs=1e-3)
