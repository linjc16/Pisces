# fmt: off

import os
import pickle
import sys
from abc import ABC, abstractmethod
from subprocess import PIPE, Popen

from ase.calculators.calculator import Calculator, all_properties


class PackedCalculator(ABC):
    """Portable calculator for use via PythonSubProcessCalculator.

    This class allows creating and talking to a calculator which
    exists inside a different process, possibly with MPI or srun.

    Use this when you want to use ASE mostly in serial, but run some
    calculations in a parallel Python environment.

    Most existing calculators can be used this way through the
    NamedPackedCalculator implementation.  To customize the behaviour
    for other calculators, write a custom class inheriting this one.

    Example::

      from ase.build import bulk

      atoms = bulk('Au')
      pack = NamedPackedCalculator('emt')

      with pack.calculator() as atoms.calc:
          energy = atoms.get_potential_energy()

    The computation takes place inside a subprocess which lives as long
    as the with statement.
    """

    @abstractmethod
    def unpack_calculator(self) -> Calculator:
        """Return the calculator packed inside.

        This method will be called inside the subprocess doing
        computations."""

    def calculator(self, mpi_command=None) -> 'PythonSubProcessCalculator':
        """Return a PythonSubProcessCalculator for this calculator.

        The subprocess calculator wraps a subprocess containing
        the actual calculator, and computations are done inside that
        subprocess."""
        return PythonSubProcessCalculator(self, mpi_command=mpi_command)


class NamedPackedCalculator(PackedCalculator):
    """PackedCalculator implementation which works with standard calculators.

    This works with calculators known by ase.calculators.calculator."""

    def __init__(self, name, kwargs=None):
        self._name = name
        if kwargs is None:
            kwargs = {}
        self._kwargs = kwargs

    def unpack_calculator(self):
        from ase.calculators.calculator import get_calculator_class
        cls = get_calculator_class(self._name)
        return cls(**self._kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._name}, {self._kwargs})'


class MPICommand:
    def __init__(self, argv):
        self.argv = argv

    @classmethod
    def python_argv(cls):
        return [sys.executable, '-m', 'ase.calculators.subprocesscalculator']

    @classmethod
    def parallel(cls, nprocs, mpi_argv=()):
        return cls(['mpiexec', '-n', str(nprocs)]
                   + list(mpi_argv)
                   + cls.python_argv()
                   + ['mpi4py'])

    @classmethod
    def serial(cls):
        return MPICommand(cls.python_argv() + ['standard'])

    def execute(self):
        # On this computer (Ubuntu 20.04 + OpenMPI) the subprocess crashes
        # without output during startup if os.environ is not passed along.
        # Hence we pass os.environ.  Not sure if this is a machine thing
        # or in general.  --askhl
        return Popen(self.argv, stdout=PIPE,
                     stdin=PIPE, env=os.environ)


def gpaw_process(ncores=1, **kwargs):
    packed = NamedPackedCalculator('gpaw', kwargs)
    mpicommand = MPICommand([
        sys.executable, '-m', 'gpaw', '-P', str(ncores), 'python', '-m',
        'ase.calculators.subprocesscalculator', 'standard',
    ])
    return PythonSubProcessCalculator(packed, mpicommand)


class PythonSubProcessCalculator(Calculator):
    """Calculator for running calculations in external processes.

    TODO: This should work with arbitrary commands including MPI stuff.

    This calculator runs a subprocess wherein it sets up an
    actual calculator.  Calculations are forwarded through pickle
    to that calculator, which returns results through pickle."""
    implemented_properties = list(all_properties)

    def __init__(self, calc_input, mpi_command=None):
        super().__init__()

        # self.proc = None
        self.calc_input = calc_input
        if mpi_command is None:
            mpi_command = MPICommand.serial()
        self.mpi_command = mpi_command

        self.protocol = None

    def set(self, **kwargs):
        if hasattr(self, 'client'):
            raise RuntimeError('No setting things for now, thanks')

    def __repr__(self):
        return '{}({})'.format(type(self).__name__,
                               self.calc_input)

    def __enter__(self):
        assert self.protocol is None
        proc = self.mpi_command.execute()
        self.protocol = Protocol(proc)
        self.protocol.send(self.calc_input)
        return self

    def __exit__(self, *args):
        self.protocol.send('stop')
        self.protocol.proc.communicate()
        self.protocol = None

    def _run_calculation(self, atoms, properties, system_changes):
        self.protocol.send('calculate')
        self.protocol.send((atoms, properties, system_changes))

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        # We send a pickle of self.atoms because this is a fresh copy
        # of the input, but without an unpicklable calculator:
        self._run_calculation(self.atoms.copy(), properties, system_changes)
        results = self.protocol.recv()
        self.results.update(results)

    def backend(self):
        return ParallelBackendInterface(self)


class Protocol:
    def __init__(self, proc):
        self.proc = proc

    def send(self, obj):
        pickle.dump(obj, self.proc.stdin)
        self.proc.stdin.flush()

    def recv(self):
        response_type, value = pickle.load(self.proc.stdout)

        if response_type == 'raise':
            raise value

        assert response_type == 'return'
        return value


class MockMethod:
    def __init__(self, name, calc):
        self.name = name
        self.calc = calc

    def __call__(self, *args, **kwargs):
        protocol = self.calc.protocol
        protocol.send('callmethod')
        protocol.send([self.name, args, kwargs])
        return protocol.recv()


class ParallelBackendInterface:
    def __init__(self, calc):
        self.calc = calc

    def __getattr__(self, name):
        return MockMethod(name, self.calc)


run_modes = {'standard', 'mpi4py'}


def callmethod(calc, attrname, args, kwargs):
    method = getattr(calc, attrname)
    value = method(*args, **kwargs)
    return value


def callfunction(func, args, kwargs):
    return func(*args, **kwargs)


def calculate(calc, atoms, properties, system_changes):
    # Again we need formalization of the results/outputs, and
    # a way to programmatically access all available properties.
    # We do a wild hack for now:
    calc.results.clear()
    # If we don't clear(), the caching is broken!  For stress.
    # But not for forces.  What dark magic from the depths of the
    # underworld is at play here?
    calc.calculate(atoms=atoms, properties=properties,
                   system_changes=system_changes)
    results = calc.results
    return results


def bad_mode():
    return SystemExit(f'sys.argv[1] must be one of {run_modes}')


def parallel_startup():
    try:
        run_mode = sys.argv[1]
    except IndexError:
        raise bad_mode()

    if run_mode not in run_modes:
        raise bad_mode()

    if run_mode == 'mpi4py':
        # We must import mpi4py before the rest of ASE, or world will not
        # be correctly initialized.
        import mpi4py  # noqa

    # We switch stdout so stray print statements won't interfere with outputs:
    binary_stdout = sys.stdout.buffer
    sys.stdout = sys.stderr

    return Client(input_fd=sys.stdin.buffer,
                  output_fd=binary_stdout)


class Client:
    def __init__(self, input_fd, output_fd):
        from ase.parallel import world
        self._world = world
        self.input_fd = input_fd
        self.output_fd = output_fd

    def recv(self):
        from ase.parallel import broadcast
        if self._world.rank == 0:
            obj = pickle.load(self.input_fd)
        else:
            obj = None

        obj = broadcast(obj, 0, self._world)
        return obj

    def send(self, obj):
        if self._world.rank == 0:
            pickle.dump(obj, self.output_fd)
            self.output_fd.flush()

    def mainloop(self, calc):
        while True:
            instruction = self.recv()
            if instruction == 'stop':
                return

            instruction_data = self.recv()

            response_type, value = self.process_instruction(
                calc, instruction, instruction_data)
            self.send((response_type, value))

    def process_instruction(self, calc, instruction, instruction_data):
        if instruction == 'callmethod':
            function = callmethod
            args = (calc, *instruction_data)
        elif instruction == 'calculate':
            function = calculate
            args = (calc, *instruction_data)
        elif instruction == 'callfunction':
            function = callfunction
            args = instruction_data
        else:
            raise RuntimeError(f'Bad instruction: {instruction}')

        try:
            value = function(*args)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            response_type = 'raise'
            value = ex
        else:
            response_type = 'return'
        return response_type, value


class ParallelDispatch:
    """Utility class to run functions in parallel.

    with ParallelDispatch(...) as parallel:
        parallel.call(function, args, kwargs)

    """

    def __init__(self, mpicommand):
        self._mpicommand = mpicommand
        self._protocol = None

    def call(self, func, *args, **kwargs):
        self._protocol.send('callfunction')
        self._protocol.send((func, args, kwargs))
        return self._protocol.recv()

    def __enter__(self):
        assert self._protocol is None
        self._protocol = Protocol(self._mpicommand.execute())

        # Even if we are not using a calculator, we have to send one:
        pack = NamedPackedCalculator('emt', {})
        self._protocol.send(pack)
        # (We should get rid of that requirement.)

        return self

    def __exit__(self, *args):
        self._protocol.send('stop')
        self._protocol.proc.communicate()
        self._protocol = None


def main():
    client = parallel_startup()
    pack = client.recv()
    calc = pack.unpack_calculator()
    client.mainloop(calc)


if __name__ == '__main__':
    main()
