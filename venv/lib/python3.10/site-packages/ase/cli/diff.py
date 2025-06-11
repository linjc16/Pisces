# fmt: off

# Note:
# Try to avoid module level import statements here to reduce
# import time during CLI execution
import sys

from ase.cli.main import CLIError

template_help = """
Without argument, looks for ~/.ase/template.py.  Otherwise,
expects the comma separated list of the fields to include
in their left-to-right order.  Optionally, specify the
lexicographical sort hierarchy (0 is outermost sort) and if the
sort should be ascending or descending (1 or -1).  By default,
sorting is descending, which makes sense for most things except
index (and rank, but one can just sort by the thing which is
ranked to get ascending ranks).

* example: ase diff start.cif stop.cif --template
* i:0:1,el,dx,dy,dz,d,rd

possible fields:

*    i: index
*    dx,dy,dz,d: displacement/displacement components
*    dfx,dfy,dfz,df: difference force/force components
*    afx,afy,afz,af: average force/force components
*    p1x,p1y,p1z,p: first image positions/position components
*    p2x,p2y,p2z,p: second image positions/position components
*    f1x,f1y,f1z,f: first image forces/force components
*    f2x,f2y,f2z,f: second image forces/force components
*    an: atomic number
*    el: atomic element
*    t: atom tag
*    r<col>: the rank of that atom with respect to the column

It is possible to change formatters in the template file."""


class CLICommand:
    """Print differences between atoms/calculations.

    Supports taking differences between different calculation runs of
    the same system as well as neighboring geometric images for one
    calculation run of a system. As part of a difference table or as a
    standalone display table, fields for non-difference quantities of image 1
    and image 2 are also provided.

    See the --template-help for the formatting exposed in the CLI.  More
    customization requires changing the input arguments to the Table
    initialization and/or editing the templates file.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('file',
            help="""Possible file entries are

    * 2 non-trajectory files: difference between them
    * 1 trajectory file: difference between consecutive images
    * 2 trajectory files: difference between corresponding image numbers
    * 1 trajectory file followed by hyphen-minus (ASCII 45): for display

    Note deltas are defined as 2 - 1.

    Use [FILE]@[SLICE] to select images.
                    """,
            nargs='+')
        add('-r',
            '--rank-order',
            metavar='FIELD',
            nargs='?',
            const='d',
            type=str,
            help="""Order atoms by rank, see --template-help for possible
fields.

The default value, when specified, is d.  When not
specified, ordering is the same as that provided by the
generator.  For hierarchical sorting, see template.""")
        add('-c', '--calculator-outputs', action="store_true",
            help="display calculator outputs of forces and energy")
        add('--max-lines', metavar='N', type=int,
            help="show only so many lines (atoms) in each table "
            ", useful if rank ordering")
        add('-t', '--template', metavar='TEMPLATE', nargs='?', const='rc',
            help="""See --template-help for the help on this option.""")
        add('--template-help', help="""Prints the help for the template file.
                Usage `ase diff - --template-help`""", action="store_true")
        add('-s', '--summary-functions', metavar='SUMFUNCS', nargs='?',
            help="""Specify the summary functions.
            Possible values are `rmsd` and `dE`.
            Comma separate more than one summary function.""")
        add('--log-file', metavar='LOGFILE', help="print table to file")
        add('--as-csv', action="store_true",
            help="output table in csv format")
        add('--precision', metavar='PREC',
            default=2, type=int,
            help="precision used in both display and sorting")

    @staticmethod
    def run(args, parser):
        import io

        if args.template_help:
            print(template_help)
            return

        encoding = 'utf-8'

        if args.log_file is None:
            out = io.TextIOWrapper(sys.stdout.buffer, encoding=encoding)
        else:
            out = open(args.log_file, 'w', encoding=encoding)

        with out:
            CLICommand.diff(args, out)

    @staticmethod
    def diff(args, out):
        from ase.cli.template import (
            Table,
            TableFormat,
            energy_delta,
            field_specs_on_conditions,
            rmsd,
            slice_split,
            summary_functions_on_conditions,
        )
        from ase.io import read

        if args.template is None:
            field_specs = field_specs_on_conditions(
                args.calculator_outputs, args.rank_order)
        else:
            field_specs = args.template.split(',')
            if not args.calculator_outputs:
                for field_spec in field_specs:
                    if 'f' in field_spec:
                        raise CLIError(
                            "field requiring calculation outputs "
                            "without --calculator-outputs")

        if args.summary_functions is None:
            summary_functions = summary_functions_on_conditions(
                args.calculator_outputs)
        else:
            summary_functions_dct = {
                'rmsd': rmsd,
                'dE': energy_delta}
            summary_functions = args.summary_functions.split(',')
            if not args.calculator_outputs:
                for sf in summary_functions:
                    if sf == 'dE':
                        raise CLIError(
                            "summary function requiring calculation outputs "
                            "without --calculator-outputs")
            summary_functions = [summary_functions_dct[i]
                                 for i in summary_functions]

        have_two_files = len(args.file) == 2
        file1 = args.file[0]
        actual_filename, index = slice_split(file1)
        atoms1 = read(actual_filename, index)
        natoms1 = len(atoms1)

        if have_two_files:
            if args.file[1] == '-':
                atoms2 = atoms1

                def header_fmt(c):
                    return f'image # {c}'
            else:
                file2 = args.file[1]
                actual_filename, index = slice_split(file2)
                atoms2 = read(actual_filename, index)
                natoms2 = len(atoms2)

                same_length = natoms1 == natoms2
                one_l_one = natoms1 == 1 or natoms2 == 1

                if not same_length and not one_l_one:
                    raise CLIError(
                        "Trajectory files are not the same length "
                        "and both > 1\n{}!={}".format(
                            natoms1, natoms2))
                elif not same_length and one_l_one:
                    print(
                        "One file contains one image "
                        "and the other multiple images,\n"
                        "assuming you want to compare all images "
                        "with one reference image")
                    if natoms1 > natoms2:
                        atoms2 = natoms1 * atoms2
                    else:
                        atoms1 = natoms2 * atoms1

                    def header_fmt(c):
                        return f'sys-ref image # {c}'
                else:
                    def header_fmt(c):
                        return f'sys2-sys1 image # {c}'
        else:
            atoms2 = atoms1.copy()
            atoms1 = atoms1[:-1]
            atoms2 = atoms2[1:]
            natoms2 = natoms1 = natoms1 - 1

            def header_fmt(c):
                return f'images {c + 1}-{c}'

        natoms = natoms1  # = natoms2

        output = ''
        tableformat = TableFormat(precision=args.precision,
                                  columnwidth=7 + args.precision)

        table = Table(
            field_specs,
            max_lines=args.max_lines,
            tableformat=tableformat,
            summary_functions=summary_functions)

        for counter in range(natoms):
            table.title = header_fmt(counter)
            output += table.make(atoms1[counter],
                                 atoms2[counter], csv=args.as_csv) + '\n'
        print(output, file=out)
