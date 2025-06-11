# fmt: off

# Note:
# Try to avoid module level import statements here to reduce
# import time during CLI execution


class CLICommand:
    """Print information about files or system.

    Without arguments, show information about ASE installation
    and library versions of dependencies.
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--files', nargs='*', metavar='PATH',
                            help='Print information about specified files.')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Show additional information about files.')
        parser.add_argument('--formats', action='store_true',
                            help='List file formats known to ASE.')
        parser.add_argument('--calculators', nargs='*', metavar='NAME',
                            help='List all or specified calculators known to '
                            'ASE and their configuration.')

    @staticmethod
    def run(args):
        if args.calculators is not None:
            from ase.codes import codes, list_codes
            if args.calculators:
                names = args.calculators
            else:
                names = [*codes]
            list_codes(names)
            return

        if args.files:
            print_file_info(args)
            return

        print_info()
        if args.formats:
            print()
            print_formats()


def print_file_info(args):
    from ase.io.bundletrajectory import print_bundletrajectory_info
    from ase.io.formats import UnknownFileTypeError, filetype, ioformats
    from ase.io.ulm import print_ulm_info
    n = max(len(filename) for filename in args.files) + 2
    nfiles_not_found = 0
    for filename in args.files:
        try:
            format = filetype(filename)
        except FileNotFoundError:
            format = '?'
            description = 'No such file'
            nfiles_not_found += 1
        except UnknownFileTypeError:
            format = '?'
            description = '?'
        else:
            if format in ioformats:
                description = ioformats[format].description
            else:
                description = '?'

        print('{:{}}{} ({})'.format(filename + ':', n,
                                    description, format))
        if args.verbose:
            if format == 'traj':
                print_ulm_info(filename)
            elif format == 'bundletrajectory':
                print_bundletrajectory_info(filename)

    raise SystemExit(nfiles_not_found)


def print_info():
    import platform
    import sys

    from ase.dependencies import all_dependencies

    versions = [('platform', platform.platform()),
                ('python-' + sys.version.split()[0], sys.executable)]

    for name, path in versions + all_dependencies():
        print(f'{name:24} {path}')


def print_formats():
    from ase.io.formats import ioformats

    print('Supported formats:')
    for fmtname in sorted(ioformats):
        fmt = ioformats[fmtname]

        infos = [fmt.modes, 'single' if fmt.single else 'multi']
        if fmt.isbinary:
            infos.append('binary')
        if fmt.encoding is not None:
            infos.append(fmt.encoding)
        infostring = '/'.join(infos)

        moreinfo = [infostring]
        if fmt.extensions:
            moreinfo.append('ext={}'.format('|'.join(fmt.extensions)))
        if fmt.globs:
            moreinfo.append('glob={}'.format('|'.join(fmt.globs)))

        print('  {} [{}]: {}'.format(fmt.name,
                                     ', '.join(moreinfo),
                                     fmt.description))
