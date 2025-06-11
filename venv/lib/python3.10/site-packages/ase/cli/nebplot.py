# fmt: off

# Note:
# Try to avoid module level import statements here to reduce
# import time during CLI execution


class CLICommand:
    """Analyze NEB trajectories by making band plots.

    One file:

        ase nebplot neb.traj

    Multiple files:

        ase nebplot neb1.traj neb2.traj

    Specify output:

        ase nebplot neb1.traj neb2.traj myfile.pdf
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('filenames', nargs='+',
            help='one or more trajectory files to analyze')
        add('output', nargs='?',
            help='optional name of output file, default=nebplots.pdf')
        add('--nimages', dest='n_images', type=int, default=None,
            help='number of images per band, guessed if not supplied')
        add('--share-x', dest='constant_x', action='store_true',
            help='use a single x axis scale for all plots')
        add('--share-y', dest='constant_y', action='store_true',
            help='use a single y axis scale for all plots')

    @staticmethod
    def run(args, parser):
        from ase.gui.images import Images
        from ase.mep import NEBTools

        # Nothing will ever be stored in args.output; need to manually find
        # if its supplied by checking extensions.
        if args.filenames[-1].endswith('.pdf'):
            args.output = args.filenames.pop(-1)
        else:
            args.output = 'nebplots.pdf'

        images = Images()
        images.read(args.filenames)
        nebtools = NEBTools(images=images)
        nebtools.plot_bands(constant_x=args.constant_x,
                            constant_y=args.constant_y,
                            nimages=args.n_images,
                            label=args.output[:-4])
