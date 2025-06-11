# fmt: off

# Note:
# Try to avoid module level import statements here to reduce
# import time during CLI execution


class CLICommand:
    """Manipulate/show content of ulm-file.

    The ULM file format is used for ASE's trajectory files,
    for GPAW's gpw-files and other things.

    Example (show first image of a trajectory file):

        ase ulm abc.traj -n 0 -v
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('filename', help='Name of ULM-file.')
        add('-n', '--index', type=int,
            help='Show only one index.  Default is to show all.')
        add('-d', '--delete', metavar='key1,key2,...',
            help='Remove key(s) from ULM-file.')
        add('-v', '--verbose', action='store_true', help='More output.')

    @staticmethod
    def run(args):
        import os

        from ase.io.ulm import copy, print_ulm_info

        if args.delete:
            exclude = {'.' + key for key in args.delete.split(',')}
            copy(args.filename, args.filename + '.temp', exclude)
            os.rename(args.filename + '.temp', args.filename)
        else:
            print_ulm_info(args.filename, args.index, verbose=args.verbose)
