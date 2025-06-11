# fmt: off

# Note:
# Try to avoid module level import statements here to reduce
# import time during CLI execution


class CLICommand:
    """Analyze the dimensionality of the bonded clusters in a structure, using
    the scoring parameter described in:

    "Definition of a scoring parameter to identify low-dimensional materials
    components",  P.M. Larsen, M. Pandey, M. Strange, and K. W. Jacobsen
    Phys. Rev. Materials 3 034003, 2019,
    https://doi.org/10.1103/PhysRevMaterials.3.034003
    https://arxiv.org/abs/1808.02114

    A score in the range [0-1] is assigned to each possible dimensionality
    classification. The scores sum to 1. A bonded cluster can be a molecular
    (0D), chain (1D), layer (2D), or bulk (3D) cluster. Mixed dimensionalities,
    such as 0D+3D are possible. Input files may use any format supported by
    ASE.

    Example usage:

    * ase dimensionality --display-all structure.cif
    * ase dimensionality structure1.cif structure2.cif

    For each structure the following data is printed:

    * type             - the dimensionalities present
    * score            - the score of the classification
    * a                - the start of the k-interval (see paper)
    * b                - the end of the k-interval (see paper)
    * component counts - the number of clusters with each dimensionality type

    If the `--display-all` option is used, all dimensionality classifications
    are displayed.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('filenames', nargs='+', help='input file(s) to analyze')
        add('--display-all', dest='full', action='store_true',
            help='display all dimensionality classifications')
        add('--no-merge', dest='no_merge', action='store_true',
            help='do not merge k-intervals with same dimensionality')

    @staticmethod
    def run(args, parser):
        import os
        import warnings

        from ase.geometry.dimensionality import analyze_dimensionality
        from ase.io import iread

        files = [os.path.split(path)[1] for path in args.filenames]
        lmax = max(len(f) for f in files) + 2

        print('file'.ljust(lmax) +
              'type   score     a      b      component counts')
        print('=' * lmax + '===============================================')

        merge = not args.no_merge

        # reading CIF files can produce a ton of distracting warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            for path, f in zip(args.filenames, files):
                for atoms in iread(path):
                    result = analyze_dimensionality(atoms, merge=merge)
                    if not args.full:
                        result = result[:1]

                    for i, entry in enumerate(result):
                        dimtype = entry.dimtype.rjust(4)
                        score = f'{entry.score:.3f}'.ljust(5)
                        a = f'{entry.a:.3f}'.ljust(5)
                        b = f'{entry.b:.3f}'.ljust(5)
                        if i == 0:
                            name = f.ljust(lmax)
                        else:
                            name = ' ' * lmax

                        line = ('{}{}' + '   {}' * 4).format(name, dimtype,
                                                             score, a, b,
                                                             entry.h)
                        print(line)

                    if args.full:
                        print()
