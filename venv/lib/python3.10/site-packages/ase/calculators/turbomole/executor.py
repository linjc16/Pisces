# fmt: off

"""
Execution of turbomole binaries and scripts:
define, dscf, grad, ridft, rdgrad, aoforce, jobex, NumForce
"""
import os
from subprocess import PIPE, Popen


def get_output_filename(basename):
    """return the output file name from the basename of the executable"""
    return 'ASE.TM.' + basename + '.out'


def check_bad_output(stderr):
    """check status written in stderr by turbomole executables"""
    if 'abnormally' in stderr or 'ended normally' not in stderr:
        raise OSError(f'Turbomole error: {stderr}')


def execute(args, input_str=''):
    """executes a turbomole executable and process the outputs"""

    stdout_file = get_output_filename(os.path.basename(args[0]))
    with open(stdout_file, 'w') as stdout:
        proc = Popen(args, stdin=PIPE, stderr=PIPE, stdout=stdout,
                     encoding='ASCII')
        _stdout_txt, stderr_txt = proc.communicate(input=input_str)
        check_bad_output(stderr_txt)
    return stdout_file
