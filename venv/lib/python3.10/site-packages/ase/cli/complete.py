#!/usr/bin/env python3
# fmt: off

"""Bash completion for ase.

Put this in your .bashrc::

    complete -o default -C /path/to/ase/cli/complete.py ase

or run::

    $ ase completion

"""

import os
import sys
from glob import glob


def match(word, *suffixes):
    return [w for w in glob(word + '*')
            if any(w.endswith(suffix) for suffix in suffixes)]


# Beginning of computer generated data:
commands = {
    'band-structure':
        ['-o', '--output', '-r', '--range'],
    'build':
        ['-M', '--magnetic-moment', '--modify', '-V', '--vacuum', '-v',
         '--vacuum0', '--unit-cell', '--bond-length', '-x',
         '--crystal-structure', '-a', '--lattice-constant',
         '--orthorhombic', '--cubic', '-r', '--repeat', '-g',
         '--gui', '--periodic'],
    'completion':
        [],
    'convert':
        ['-v', '--verbose', '-i', '--input-format', '-o',
         '--output-format', '-f', '--force', '-n',
         '--image-number', '-e', '--exec-code', '-E',
         '--exec-file', '-a', '--arrays', '-I', '--info', '-s',
         '--split-output', '--read-args', '--write-args'],
    'db':
        ['-v', '--verbose', '-q', '--quiet', '-n', '--count', '-l',
         '--long', '-i', '--insert-into', '-a',
         '--add-from-file', '-k', '--add-key-value-pairs', '-L',
         '--limit', '--offset', '--delete', '--delete-keys',
         '-y', '--yes', '--explain', '-c', '--columns', '-s',
         '--sort', '--cut', '-p', '--plot', '--csv', '-w',
         '--open-web-browser', '--no-lock-file', '--analyse',
         '-j', '--json', '-m', '--show-metadata',
         '--set-metadata', '--strip-data', '--progress-bar',
         '--show-keys', '--show-values'],
    'diff':
        ['-r', '--rank-order', '-c', '--calculator-outputs',
         '--max-lines', '-t', '--template', '--template-help',
         '-s', '--summary-functions', '--log-file', '--as-csv',
         '--precision'],
    'dimensionality':
        ['--display-all', '--no-merge'],
    'eos':
        ['-p', '--plot', '-t', '--type'],
    'exec':
        ['-e', '--exec-code', '-E', '--exec-file', '-i', '--input-format',
         '-n', '--image-number', '--read-args'],
    'find':
        ['-v', '--verbose', '-l', '--long', '-i', '--include', '-x',
         '--exclude'],
    'gui':
        ['-n', '--image-number', '-r', '--repeat', '-R', '--rotations',
         '-o', '--output', '-g', '--graph', '-t', '--terminal',
         '--interpolate', '-b', '--bonds', '-s', '--scale'],
    'info':
        ['--files', '-v', '--verbose', '--formats', '--calculators'],
    'nebplot':
        ['--nimages', '--share-x', '--share-y'],
    'reciprocal':
        [],
    'run':
        ['-p', '--parameters', '-t', '--tag', '--properties', '-f',
         '--maximum-force', '--constrain-tags', '-s',
         '--maximum-stress', '-E', '--equation-of-state',
         '--eos-type', '-o', '--output', '--modify', '--after'],
    'test':
        ['-c', '--calculators', '--help-calculators', '--list',
         '--list-calculators', '-j', '--jobs', '-v', '--verbose',
         '--strict', '--fast', '--coverage', '--nogui',
         '--pytest'],
    'ulm':
        ['-n', '--index', '-d', '--delete', '-v', '--verbose']}
# End of computer generated data


def complete(word, previous, line, point):
    for w in line[:point - len(word)].strip().split()[1:]:
        if w[0].isalpha():
            if w in commands:
                command = w
                break
    else:
        if word[:1] == '-':
            return ['-h', '--help', '--version']
        return list(commands.keys()) + ['-h', '--help', '--verbose']

    if word[:1] == '-':
        return commands[command]

    words = []

    if command == 'db':
        if previous == 'db':
            words = match(word, '.db', '.json')

    elif command == 'run':
        if previous == 'run':
            from ase.calculators.calculator import names as words

    elif command == 'build':
        if previous in ['-x', '--crystal-structure']:
            words = ['sc', 'fcc', 'bcc', 'hcp', 'diamond', 'zincblende',
                     'rocksalt', 'cesiumchloride', 'fluorite', 'wurtzite']

    elif command == 'test':
        if previous in ['-c', '--calculators']:
            from ase.calculators.calculator import names as words
        elif not word.startswith('-'):
            from ase.test.testsuite import all_test_modules_and_groups
            words = []
            for path in all_test_modules_and_groups():
                path = str(path)
                if not path.endswith('.py'):
                    path += '/'
                words.append(path)

    return words


def main():
    word, previous = sys.argv[2:]
    line = os.environ['COMP_LINE']
    point = int(os.environ['COMP_POINT'])
    words = complete(word, previous, line, point)
    for w in words:
        if w.startswith(word):
            print(w)


if __name__ == '__main__':
    main()
