# fmt: off

"""TAB-completion sub-command and update helper funtion.

Run this when ever options are changed::

    python3 -m ase.cli.completion
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Path of the complete.py script:
path = Path(__file__).with_name('complete.py')


class CLICommand:
    """Add tab-completion for Bash.

    Will show the command that needs to be added to your '~/.bashrc file.
    """
    cmd = f'complete -o default -C "{sys.executable} {path}" ase'

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args):
        cmd = CLICommand.cmd
        print(cmd)


def update(path: Path,
           subcommands: List[Tuple[str, str]],
           test: bool = False) -> None:
    """Update commands dict in complete.py.

    Use test=True to test that no changes are needed.

    Refactor with care!  This function is also used by GPAW.
    """

    import textwrap
    from importlib import import_module

    dct: Dict[str, List[str]] = {}

    class Subparser:
        def __init__(self, command):
            self.command = command
            dct[command] = []

        def add_argument(self, *args, **kwargs):
            dct[command].extend(arg for arg in args
                                if arg.startswith('-'))

        def add_argument_group(self, name):
            return self

        def add_mutually_exclusive_group(self, required=False):
            return self

    for command, module_name in subcommands:
        module = import_module(module_name)
        module.CLICommand.add_arguments(Subparser(command))

    txt = 'commands = {'
    for command, opts in sorted(dct.items()):
        txt += "\n    '" + command + "':\n        ["
        if opts:
            txt += '\n'.join(textwrap.wrap("'" + "', '".join(opts) + "'],",
                                           width=65,
                                           break_on_hyphens=False,
                                           subsequent_indent='         '))
        else:
            txt += '],'
    txt = txt[:-1] + '}\n'

    with path.open() as fd:
        lines = fd.readlines()

    a = lines.index('# Beginning of computer generated data:\n')
    b = lines.index('# End of computer generated data\n')

    if test:
        if ''.join(lines[a + 1:b]) != txt:
            raise ValueError(
                'Please update ase/cli/complete.py using '
                '"python3 -m ase.cli.completion".')
    else:
        lines[a + 1:b] = [txt]
        new = path.with_name('complete.py.new')
        with new.open('w') as fd:
            print(''.join(lines), end='', file=fd)
        new.rename(path)
        path.chmod(0o775)


if __name__ == '__main__':
    from ase.cli.main import commands
    update(path, commands)
