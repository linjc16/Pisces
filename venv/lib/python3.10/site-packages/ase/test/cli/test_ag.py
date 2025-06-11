from ase import Atoms
from ase.io import write


def test_ag(cli, testdir):
    write('x.json', Atoms('X'))

    # Make sure ASE's gui can run in terminal mode without $DISPLAY and tkinter:
    cli.shell('ase -T gui --terminal -n "id=1" x.json')
