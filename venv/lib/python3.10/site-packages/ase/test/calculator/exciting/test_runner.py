"""Test runner classes to run exciting simulations using subproces."""

import pytest

import ase.calculators.exciting.runner


@pytest.mark.parametrize(
    (
        "binary, run_argv, omp_num_threads,"), [
            ("exciting_serial", ['./'], 1),
            ("exciting_purempi", ['mpirun', '-np', '1'], 1),
            ("exciting_smp", ['./'], 1),
            ("exciting_mpismp", ['mpirun', '-np', '1'], 1)
    ])
def test_class_simple_binary_runner_init(
        tmpdir,
        binary,
        run_argv,
        omp_num_threads,
        excitingtools):
    """Test SimpleBinaryRunner."""
    binary = tmpdir / 'binary.exe'
    binary.write("Arbitrary text such that file exists")
    runner = ase.calculators.exciting.runner.SimpleBinaryRunner(
        binary=binary, run_argv=run_argv,
        omp_num_threads=omp_num_threads,
        directory=tmpdir,
        args=[])

    # Attributes
    assert runner.binary == binary, "Binary erroneously initialised"
    assert runner.run_argv == run_argv
    assert runner.omp_num_threads == omp_num_threads
    assert runner.directory == tmpdir
    assert runner.args == []


def test_compose_execution_list(tmpdir):
    """Test SimpleBinaryRunner's compose execution list method."""
    binary = tmpdir / 'binary.exe'
    binary.write("Arbitrary text such that file exists")
    runner = ase.calculators.exciting.runner.SimpleBinaryRunner(
        binary=binary, run_argv=['mpirun', '-np', '2'], omp_num_threads=1,
        directory=tmpdir,
        args=['input.txt'])
    # Attributes
    assert runner.binary == binary
    assert runner.run_argv == ['mpirun', '-np', '2']
    assert runner.omp_num_threads == 1
    assert runner.directory == tmpdir
    assert runner.args == ['input.txt']
    # Methods
    execute = runner.compose_execution_list()
    assert execute == ['mpirun', '-np', '2', str(binary), 'input.txt']
