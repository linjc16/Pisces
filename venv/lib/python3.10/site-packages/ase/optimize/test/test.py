# fmt: off

import argparse
import traceback
from math import pi
from time import time

import numpy as np

import ase.db
import ase.optimize
from ase.calculators.emt import EMT
from ase.io import Trajectory

all_optimizers = ase.optimize.__all__ + ['PreconLBFGS', 'PreconFIRE',
                                         'SciPyFminCG', 'SciPyFminBFGS']
all_optimizers.remove('QuasiNewton')
all_optimizers.remove('RestartError')


def get_optimizer(name):
    # types: (str) -> ase.optimize.Optimizer
    if name.startswith('Precon'):
        import ase.optimize.precon as precon
        return getattr(precon, name)
    if name.startswith('SciPy'):
        import ase.optimize.sciopt as sciopt
        return getattr(sciopt, name)
    return getattr(ase.optimize, name)


class Wrapper:
    """Atoms-object wrapper that can count number of moves."""

    def __init__(self, atoms, gridspacing=0.2, eggbox=0.0):
        # types: (Atoms, float, float) -> None
        self.t0 = time()
        self.texcl = 0.0
        self.nsteps = 0
        self.atoms = atoms
        self.ready = False
        self.pos = None  # type: np.ndarray
        self.eggbox = eggbox

        if eggbox:
            # Find small unit cell for grid-points
            h = []
            for axis in atoms.get_cell(complete=True):
                L = np.linalg.norm(axis)
                n = int(L / gridspacing)
                h.append(axis / n)
            self.x = np.linalg.inv(h)
        else:
            self.x = None

    def get_potential_energy(self, force_consistent=False):
        t1 = time()
        e = self.atoms.get_potential_energy(force_consistent)

        if self.eggbox:
            # Add egg-box error:
            s = np.dot(self.atoms.positions, self.x)
            e += np.cos(2 * pi * s).sum() * self.eggbox / 6

        t2 = time()
        self.texcl += t2 - t1
        if not self.ready:
            self.nsteps += 1
        self.ready = True
        return e

    def get_forces(self):
        t1 = time()
        f = self.atoms.get_forces()

        if self.eggbox:
            # Add egg-box error:
            s = np.dot(self.atoms.positions, self.x)
            f += np.dot(np.sin(2 * pi * s),
                        self.x.T) * (2 * pi * self.eggbox / 6)

        t2 = time()
        self.texcl += t2 - t1
        if not self.ready:
            self.nsteps += 1
        self.ready = True
        return f

    def set_positions(self, pos):
        if self.pos is not None and abs(pos - self.pos).max() > 1e-15:
            self.ready = False
            if self.nsteps == 200:
                raise RuntimeError('Did not converge!')

        self.pos = pos
        self.atoms.set_positions(pos)

    def get_positions(self):
        return self.atoms.get_positions()

    def get_calculator(self):
        return self.atoms.calc

    def __len__(self):
        return len(self.atoms)

    def __ase_optimizable__(self):
        from ase.optimize.optimize import OptimizableAtoms
        return OptimizableAtoms(self)


def run_test(atoms, optimizer, tag, fmax=0.02, eggbox=0.0):
    """Optimize atoms with optimizer."""
    wrapper = Wrapper(atoms, eggbox=eggbox)
    relax = optimizer(wrapper, logfile=tag + '.log')
    relax.attach(Trajectory(tag + '.traj', 'w', atoms=atoms))

    tincl = -time()
    error = ''

    try:
        relax.run(fmax=fmax, steps=10000000)
    except Exception as x:
        wrapper.nsteps = float('inf')
        error = f'{x.__class__.__name__}: {x}'
        tb = traceback.format_exc()

        with open(tag + '.err', 'w') as fd:
            fd.write(f'{error}\n{tb}\n')

    tincl += time()

    return error, wrapper.nsteps, wrapper.texcl, tincl


def test_optimizer(systems, optimizer, calculator, prefix='', db=None,
                   eggbox=0.0):
    """Test optimizer on systems."""

    for name, atoms in systems:
        if db is not None:
            optname = optimizer.__name__
            id = db.reserve(optimizer=optname, name=name)
            if id is None:
                continue
        atoms = atoms.copy()
        tag = f'{prefix}{optname}-{name}'
        atoms.calc = calculator(txt=tag + '.txt')
        error, nsteps, texcl, tincl = run_test(atoms, optimizer, tag,
                                               eggbox=eggbox)

        if db is not None:
            db.write(atoms,
                     id=id,
                     optimizer=optname,
                     name=name,
                     error=error,
                     n=nsteps,
                     t=texcl,
                     T=tincl,
                     eggbox=eggbox)


def main():
    parser = argparse.ArgumentParser(
        description='Test ASE optimizers')

    parser.add_argument('systems', help='File containing test systems.')
    parser.add_argument('optimizer', nargs='*',
                        help='Optimizer name(s).  Choose from: {}. '
                        .format(', '.join(all_optimizers)) +
                        'Default is all optimizers.')
    parser.add_argument('-e', '--egg-box', type=float, default=0.0,
                        help='Fake egg-box error in eV.')

    args = parser.parse_args()

    systems = [(row.name, row.toatoms())
               for row in ase.db.connect(args.systems).select()]

    db = ase.db.connect('results.db')

    if not args.optimizer:
        args.optimizer = all_optimizers

    for opt in args.optimizer:
        print(opt)
        optimizer = get_optimizer(opt)
        test_optimizer(systems, optimizer, EMT, db=db, eggbox=args.egg_box)


if __name__ == '__main__':
    main()
