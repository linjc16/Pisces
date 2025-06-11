# fmt: off

"""Binary runner and results class."""
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Union


class SubprocessRunResults:
    """Results returned from subprocess.run()."""

    def __init__(
            self, stdout, stderr, return_code: int,
            process_time: Optional[float] = None):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code
        self.success = return_code == 0
        self.process_time = process_time


class SimpleBinaryRunner:
    """Class to execute a subprocess."""
    path_type = Union[str, Path]

    def __init__(self,
                 binary,
                 run_argv: List[str],
                 omp_num_threads: int,
                 directory: path_type = './',
                 args=None) -> None:
        """Initialise class.

        :param binary: Binary name prepended by full path, or just binary name
            (if present in $PATH).
        :param run_argv: Run commands sequentially as a list of str.
            For example:
            * For serial: ['./'] or ['']
            * For MPI:   ['mpirun', '-np', '2']
        :param omp_num_threads: Number of OMP threads.
        :param args: Optional arguments for the binary.
        """
        if args is None:
            args = []
        self.binary = binary
        self.directory = directory

        self.run_argv = run_argv

        self.omp_num_threads = omp_num_threads
        self.args = args

        if directory is not None and not Path(directory).is_dir():
            raise OSError(f"Run directory does not exist: {directory}")

        if omp_num_threads <= 0:
            raise ValueError("Number of OMP threads must be > 0")

    def compose_execution_list(self) -> list:
        """Generate a complete list of strings to pass to subprocess.run().

        This is done to execute the calculation.

        For example, given:
          ['mpirun', '-np, '2'] + ['binary.exe'] + ['>', 'std.out']

        return ['mpirun', '-np, '2', 'binary.exe', '>', 'std.out']
        """
        return self.run_argv + [self.binary] + self.args

    def run(self) -> SubprocessRunResults:
        """Run a binary."""
        execution_list = self.compose_execution_list()
        my_env = {**os.environ}

        time_start: float = time.time()
        result = subprocess.run(execution_list,
                                env=my_env,
                                capture_output=True,
                                cwd=self.directory, check=False)
        total_time = time.time() - time_start
        return SubprocessRunResults(
            result.stdout, result.stderr, result.returncode, total_time)
