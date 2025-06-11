# fmt: off

# Note:
# Try to avoid module level import statements here to reduce
# import time during CLI execution
from ase.cli.main import CLIError


def plot_reciprocal_cell(path, output=None):
    import matplotlib.pyplot as plt

    path.plot()

    if output:
        plt.savefig(output)
    else:
        plt.show()


def read_object(filename):
    from ase.io import read
    from ase.io.formats import UnknownFileTypeError
    from ase.io.jsonio import read_json

    try:
        return read(filename)
    except UnknownFileTypeError:
        # Probably a bandpath/bandstructure:
        return read_json(filename)


def obj2bandpath(obj):
    from ase import Atoms
    from ase.dft.kpoints import BandPath

    if isinstance(obj, BandPath):
        print('Object is a band path')
        print(obj)
        return obj

    if isinstance(getattr(obj, 'path', None), BandPath):
        print(f'Object contains a bandpath: {obj}')
        path = obj.path
        print(path)
        return path

    if isinstance(obj, Atoms):
        print(f'Atoms object: {obj}')
        print('Determining standard form of Bravais lattice:')
        lat = obj.cell.get_bravais_lattice(pbc=obj.pbc)
        print(lat.description())
        print('Showing default bandpath')
        return lat.bandpath(density=0)

    raise CLIError(f'Strange object: {obj}')


class CLICommand:
    """Show the reciprocal space.

    Read unit cell from a file and show a plot of the 1. Brillouin zone.  If
    the file contains information about k-points, then those can be plotted
    too.

    Examples:

        $ ase build -x fcc Al al.traj
        $ ase reciprocal al.traj
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('name', metavar='input-file',
            help='Input file containing unit cell.')
        add('output', nargs='?', help='Write plot to file (.png, .svg, ...).')

    @staticmethod
    def run(args, parser):
        obj = read_object(args.name)
        path = obj2bandpath(obj)
        plot_reciprocal_cell(path, output=args.output)
