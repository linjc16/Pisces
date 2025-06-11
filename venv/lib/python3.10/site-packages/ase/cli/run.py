# fmt: off

import sys
from typing import Any, Dict

import numpy as np


class CLICommand:
    """Run calculation with one of ASE's calculators.

    Four types of calculations can be done:

    * single point
    * atomic relaxations
    * unit cell + atomic relaxations
    * equation-of-state

    Examples of the four types of calculations:

        ase run emt h2o.xyz
        ase run emt h2o.xyz -f 0.01
        ase run emt cu.traj -s 0.01
        ase run emt cu.traj -E 5,2.0
    """

    @staticmethod
    def add_arguments(parser):
        from ase.calculators.names import names
        parser.add_argument('calculator',
                            help='Name of calculator to use.  '
                            'Must be one of: {}.'
                            .format(', '.join(names)))
        CLICommand.add_more_arguments(parser)

    @staticmethod
    def add_more_arguments(parser):
        add = parser.add_argument
        add('name', nargs='?', default='-',
            help='Read atomic structure from this file.')
        add('-p', '--parameters', default='',
            metavar='key=value,...',
            help='Comma-separated key=value pairs of ' +
            'calculator specific parameters.')
        add('-t', '--tag',
            help='String tag added to filenames.')
        add('--properties', default='efsdMm',
            help='Default value is "efsdMm" meaning calculate energy, ' +
            'forces, stress, dipole moment, total magnetic moment and ' +
            'atomic magnetic moments.')
        add('-f', '--maximum-force', type=float,
            help='Relax internal coordinates.')
        add('--constrain-tags',
            metavar='T1,T2,...',
            help='Constrain atoms with tags T1, T2, ...')
        add('-s', '--maximum-stress', type=float,
            help='Relax unit-cell and internal coordinates.')
        add('-E', '--equation-of-state',
            help='Use "-E 5,2.0" for 5 lattice constants ranging from '
            '-2.0 %% to +2.0 %%.')
        add('--eos-type', default='sjeos', help='Selects the type of eos.')
        add('-o', '--output', help='Write result to file (append mode).')
        add('--modify', metavar='...',
            help='Modify atoms with Python statement.  ' +
            'Example: --modify="atoms.positions[-1,2]+=0.1".')
        add('--after', help='Perform operation after calculation.  ' +
            'Example: --after="atoms.calc.write(...)"')

    @staticmethod
    def run(args):
        runner = Runner()
        runner.parse(args)
        runner.run()


class Runner:
    def __init__(self):
        self.args = None
        self.calculator_name = None

    def parse(self, args):
        self.calculator_name = args.calculator
        self.args = args

    def run(self):
        args = self.args

        atoms = self.build(args.name)
        if args.modify:
            exec(args.modify, {'atoms': atoms, 'np': np})

        if args.name == '-':
            args.name = 'stdin'

        self.set_calculator(atoms, args.name)

        self.calculate(atoms, args.name)

    def calculate(self, atoms, name):
        from ase.io import write

        args = self.args

        if args.maximum_force or args.maximum_stress:
            self.optimize(atoms, name)
        if args.equation_of_state:
            self.eos(atoms, name)
        self.calculate_once(atoms)

        if args.after:
            exec(args.after, {'atoms': atoms})

        if args.output:
            write(args.output, atoms, append=True)

    def build(self, name):
        import ase.db as db
        from ase.io import read

        if name == '-':
            con = db.connect(sys.stdin, 'json')
            return con.get_atoms(add_additional_information=True)
        else:
            atoms = read(name)
            if isinstance(atoms, list):
                assert len(atoms) == 1
                atoms = atoms[0]
            return atoms

    def set_calculator(self, atoms, name):
        from ase.calculators.calculator import get_calculator_class

        cls = get_calculator_class(self.calculator_name)
        parameters = str2dict(self.args.parameters)
        if getattr(cls, 'nolabel', False):
            atoms.calc = cls(**parameters)
        else:
            atoms.calc = cls(label=self.get_filename(name), **parameters)

    def calculate_once(self, atoms):
        from ase.calculators.calculator import PropertyNotImplementedError

        args = self.args

        for p in args.properties or 'efsdMm':
            _property, method = {'e': ('energy', 'get_potential_energy'),
                                'f': ('forces', 'get_forces'),
                                's': ('stress', 'get_stress'),
                                'd': ('dipole', 'get_dipole_moment'),
                                'M': ('magmom', 'get_magnetic_moment'),
                                'm': ('magmoms', 'get_magnetic_moments')}[p]
            try:
                getattr(atoms, method)()
            except PropertyNotImplementedError:
                pass

    def optimize(self, atoms, name):
        from ase.constraints import FixAtoms, UnitCellFilter
        from ase.io import Trajectory
        from ase.optimize import LBFGS

        args = self.args
        if args.constrain_tags:
            tags = [int(t) for t in args.constrain_tags.split(',')]
            mask = [t in tags for t in atoms.get_tags()]
            atoms.constraints = FixAtoms(mask=mask)

        logfile = self.get_filename(name, 'log')
        if args.maximum_stress:
            optimizer = LBFGS(UnitCellFilter(atoms), logfile=logfile)
            fmax = args.maximum_stress
        else:
            optimizer = LBFGS(atoms, logfile=logfile)
            fmax = args.maximum_force

        trajectory = Trajectory(self.get_filename(name, 'traj'), 'w', atoms)
        optimizer.attach(trajectory)
        optimizer.run(fmax=fmax)

    def eos(self, atoms, name):
        from ase.eos import EquationOfState
        from ase.io import Trajectory

        args = self.args

        traj = Trajectory(self.get_filename(name, 'traj'), 'w', atoms)

        N, eps = args.equation_of_state.split(',')
        N = int(N)
        eps = float(eps) / 100
        strains = np.linspace(1 - eps, 1 + eps, N)
        v1 = atoms.get_volume()
        volumes = strains**3 * v1
        energies = []
        cell1 = atoms.cell.copy()
        for s in strains:
            atoms.set_cell(cell1 * s, scale_atoms=True)
            energies.append(atoms.get_potential_energy())
            traj.write(atoms)
        traj.close()
        eos = EquationOfState(volumes, energies, args.eos_type)
        v0, e0, B = eos.fit()
        atoms.set_cell(cell1 * (v0 / v1)**(1 / 3), scale_atoms=True)
        from ase.parallel import parprint as p
        p('volumes:', volumes)
        p('energies:', energies)
        p('fitted energy:', e0)
        p('fitted volume:', v0)
        p('bulk modulus:', B)
        p('eos type:', args.eos_type)

    def get_filename(self, name: str, ext: str = '') -> str:
        if '.' in name:
            name = name.rsplit('.', 1)[0]
        if self.args.tag is not None:
            name += '-' + self.args.tag
        if ext:
            name += '.' + ext
        return name


def str2dict(s: str, namespace={}, sep: str = '=') -> Dict[str, Any]:
    """Convert comma-separated key=value string to dictionary.

    Examples:

    >>> str2dict('xc=PBE,nbands=200,parallel={band:4}')
    {'xc': 'PBE', 'nbands': 200, 'parallel': {'band': 4}}
    >>> str2dict('a=1.2,b=True,c=ab,d=1,2,3,e={f:42,g:cd}')
    {'a': 1.2, 'c': 'ab', 'b': True, 'e': {'g': 'cd', 'f': 42}, 'd': (1, 2, 3)}
    """

    def myeval(value):
        try:
            value = eval(value, namespace)
        except (NameError, SyntaxError):
            pass
        return value

    dct = {}
    strings = (s + ',').split(sep)
    for i in range(len(strings) - 1):
        key = strings[i]
        m = strings[i + 1].rfind(',')
        value: Any = strings[i + 1][:m]
        if value[0] == '{':
            assert value[-1] == '}'
            value = str2dict(value[1:-1], namespace, ':')
        elif value[0] == '(':
            assert value[-1] == ')'
            value = [myeval(t) for t in value[1:-1].split(',')]
        else:
            value = myeval(value)
        dct[key] = value
        strings[i + 1] = strings[i + 1][m + 1:]
    return dct
