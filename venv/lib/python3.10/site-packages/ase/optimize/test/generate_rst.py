# fmt: off

import os
import re

rst = """\
.. _optimizer_tests:

===============
Optimizer tests
===============
This page shows benchmarks of optimizations done with our different optimizers.
Note that the iteration number (steps) is not the same as the number of force
evaluations. This is because some of the optimizers uses internal line searches
or similar.

The most important performance characteristics of an optimizer is the
total optimization time.
Different optimizers may perform the same number of steps, but along a different
path, so the time spent on calculation of energy/forces will be different.
"""


header = 'Optimizer       Steps Force evaluations Energy     Note           \n'
bars = '=============== ===== ================= ========== ===============\n'


def main():
    dirlist = os.listdir('.')
    name = r'.*\.csv'
    filterre = re.compile(name)
    dirlist = list(filter(filterre.search, dirlist))
    namelist = [d.strip('.csv') for d in dirlist]

    with open('testoptimize.rst', 'w') as fd:
        fd.write(rst)

        for name in namelist:
            lines = open(name + '.csv').read().split('\n')
            firstline = lines.pop(0)
            fd.write(
                '\n' +
                name + '\n' +
                '=' * len(name) + '\n'
                'Calculator used: %s\n' % firstline.split(',')[-1] +
                '\n' +
                bars +
                header +
                bars
            )
            for line in lines:
                if len(line):
                    print(line.split(','))
                    fd.write(
                        '%-15s %5s %17s %10s %s\n' % tuple(line.split(','))
                    )
            fd.write(
                bars
            )


if __name__ == '__main__':
    main()
