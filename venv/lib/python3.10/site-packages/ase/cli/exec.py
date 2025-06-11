# fmt: off

# Note:
# Try to avoid module level import statements here to reduce
# import time during CLI execution


class CLICommand:
    """ Execute code on files.

    The given python code is evaluated on the Atoms object read from
    the input file for each frame of the file. Either of -e or -E
    option should provided for evaluating code given as a string or
    from a file, respectively.

    Variables which can be used inside the python code:
    - `index`: Index of the current Atoms object.
    - `atoms`: Current Atoms object.
    - `images`: List of all images given as input.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('input', nargs='+', metavar='input-file')
        add('-e', '--exec-code',
            help='Python code to execute on each atoms. The Atoms'
            ' object is available as `atoms`. '
            'Example: For printing cell parameters from all the '
            'frames, `print(atoms.cell.cellpar())`')
        add('-E', '--exec-file',
            help='Python source code file to execute on each '
            'frame, usage is as for -e/--exec-code.')
        add('-i', '--input-format', metavar='FORMAT',
            help='Specify input FORMAT')
        add('-n', '--image-number',
            default=':', metavar='NUMBER',
            help='Pick images from trajectory.  NUMBER can be a '
            'single number (use a negative number to count from '
            'the back) or a range: start:stop:step, where the '
            '":step" part can be left out - default values are '
            '0:nimages:1.')
        add('--read-args', nargs='+', action='store',
            default={}, metavar="KEY=VALUE",
            help='Additional keyword arguments to pass to '
            '`ase.io.read()`.')

    @staticmethod
    def run(args, parser):
        import runpy

        from ase.io import read

        if not (args.exec_code or args.exec_file):
            parser.error("At least one of '-e' or '-E' must be provided")

        if args.read_args:
            args.read_args = eval("dict({})"
                                  .format(', '.join(args.read_args)))

        configs = []
        for filename in args.input:
            atoms = read(filename, args.image_number,
                         format=args.input_format, **args.read_args)
            if isinstance(atoms, list):
                configs.extend(atoms)
            else:
                configs.append(atoms)

        variables = {'images': configs}
        for index, atoms in enumerate(configs):
            variables['atoms'] = atoms
            variables['index'] = index
            if args.exec_code:
                exec(compile(args.exec_code, '<string>', 'exec'), variables)
            if args.exec_file:
                runpy.run_path(args.exec_file, init_globals=variables)
