# fmt: off

import string

import numpy as np

from ase.data import chemical_symbols
from ase.io import string2index
from ase.io.formats import parse_filename

# default fields


def field_specs_on_conditions(calculator_outputs, rank_order):
    if calculator_outputs:
        field_specs = ['i:0', 'el', 'd', 'rd', 'df', 'rdf']
    else:
        field_specs = ['i:0', 'el', 'dx', 'dy', 'dz', 'd', 'rd']
    if rank_order is not None:
        field_specs[0] = 'i:1'
        if rank_order in field_specs:
            for c, i in enumerate(field_specs):
                if i == rank_order:
                    field_specs[c] = i + ':0:1'
        else:
            field_specs.append(rank_order + ':0:1')
    else:
        field_specs[0] = field_specs[0] + ':1'
    return field_specs


def summary_functions_on_conditions(has_calc):
    if has_calc:
        return [rmsd, energy_delta]
    return [rmsd]


def header_alias(h):
    """Replace keyboard characters with Unicode symbols
    for pretty printing"""
    if h == 'i':
        h = 'index'
    elif h == 'an':
        h = 'atomic #'
    elif h == 't':
        h = 'tag'
    elif h == 'el':
        h = 'element'
    elif h[0] == 'd':
        h = h.replace('d', 'Δ')
    elif h[0] == 'r':
        h = 'rank ' + header_alias(h[1:])
    elif h[0] == 'a':
        h = h.replace('a', '<')
        h += '>'
    return h


def prec_round(a, prec=2):
    """
    To make hierarchical sorting different from non-hierarchical sorting
    with floats.
    """
    if a == 0:
        return a
    else:
        s = 1 if a > 0 else -1
        m = np.log10(s * a) // 1
        c = np.log10(s * a) % 1
    return s * np.round(10**c, prec) * 10**m


prec_round = np.vectorize(prec_round)

# end most settings

# this will sort alphabetically by chemical symbol
num2sym = dict(zip(np.argsort(chemical_symbols), chemical_symbols))
# to sort by atomic number, uncomment below
# num2sym = dict(zip(range(len(chemical_symbols)), chemical_symbols))
sym2num = {v: k for k, v in num2sym.items()}

atoms_props = [
    'dx',
    'dy',
    'dz',
    'd',
    't',
    'an',
    'i',
    'el',
    'p1',
    'p2',
    'p1x',
    'p1y',
    'p1z',
    'p2x',
    'p2y',
    'p2z']


def get_field_data(atoms1, atoms2, field):
    if field[0] == 'r':
        field = field[1:]
        rank_order = True
    else:
        rank_order = False

    if field in atoms_props:
        if field == 't':
            data = atoms1.get_tags()
        elif field == 'an':
            data = atoms1.numbers
        elif field == 'el':
            data = np.array([sym2num[sym] for sym in atoms1.symbols])
        elif field == 'i':
            data = np.arange(len(atoms1))
        else:
            if field.startswith('d'):
                y = atoms2.positions - atoms1.positions
            elif field.startswith('p'):
                if field[1] == '1':
                    y = atoms1.positions
                else:
                    y = atoms2.positions

            if field.endswith('x'):
                data = y[:, 0]
            elif field.endswith('y'):
                data = y[:, 1]
            elif field.endswith('z'):
                data = y[:, 2]
            else:
                data = np.linalg.norm(y, axis=1)
    else:
        if field[0] == 'd':
            y = atoms2.get_forces() - atoms1.get_forces()
        elif field[0] == 'a':
            y = (atoms2.get_forces() + atoms1.get_forces()) / 2
        else:
            if field[1] == '1':
                y = atoms1.get_forces()
            else:
                y = atoms2.get_forces()

        if field.endswith('x'):
            data = y[:, 0]
        elif field.endswith('y'):
            data = y[:, 1]
        elif field.endswith('z'):
            data = y[:, 2]
        else:
            data = np.linalg.norm(y, axis=1)

    if rank_order:
        return np.argsort(np.argsort(-data))

    return data


# Summary Functions

def rmsd(atoms1, atoms2):
    dpositions = atoms2.positions - atoms1.positions
    return 'RMSD={:+.1E}'.format(
        np.sqrt((np.linalg.norm(dpositions, axis=1)**2).mean()))


def energy_delta(atoms1, atoms2):
    E1 = atoms1.get_potential_energy()
    E2 = atoms2.get_potential_energy()
    return f'E1 = {E1:+.1E}, E2 = {E2:+.1E}, dE = {E2 - E1:+1.1E}'


def parse_field_specs(field_specs):
    fields = []
    hier = []
    scent = []
    for fs in field_specs:
        fhs = fs.split(':')
        if len(fhs) == 3:
            scent.append(int(fhs[2]))
            hier.append(int(fhs[1]))
            fields.append(fhs[0])
        elif len(fhs) == 2:
            scent.append(-1)
            hier.append(int(fhs[1]))
            fields.append(fhs[0])
        elif len(fhs) == 1:
            scent.append(-1)
            hier.append(-1)
            fields.append(fhs[0])
    mxm = max(hier)
    for c in range(len(hier)):
        if hier[c] < 0:
            mxm += 1
            hier[c] = mxm
    # reversed by convention of numpy lexsort
    hier = np.argsort(hier)[::-1]
    return fields, hier, np.array(scent)

# Class definitions


class MapFormatter(string.Formatter):
    """String formatting method to map string
    mapped to float data field
    used for sorting back to string."""

    def format_field(self, value, spec):
        if spec.endswith('h'):
            value = num2sym[int(value)]
            spec = spec[:-1] + 's'
        return super().format_field(value, spec)


class TableFormat:
    def __init__(self,
                 columnwidth=9,
                 precision=2,
                 representation='E',
                 toprule='=',
                 midrule='-',
                 bottomrule='='):

        self.precision = precision
        self.representation = representation
        self.columnwidth = columnwidth
        self.formatter = MapFormatter().format
        self.toprule = toprule
        self.midrule = midrule
        self.bottomrule = bottomrule

        self.fmt_class = {
            'signed float': "{{: ^{}.{}{}}}".format(
                self.columnwidth,
                self.precision - 1,
                self.representation),
            'unsigned float': "{{:^{}.{}{}}}".format(
                self.columnwidth,
                self.precision - 1,
                self.representation),
            'int': "{{:^{}n}}".format(
                self.columnwidth),
            'str': "{{:^{}s}}".format(
                self.columnwidth),
            'conv': "{{:^{}h}}".format(
                self.columnwidth)}
        fmt = {}
        signed_floats = [
            'dx',
            'dy',
            'dz',
            'dfx',
            'dfy',
            'dfz',
            'afx',
            'afy',
            'afz',
            'p1x',
            'p2x',
            'p1y',
            'p2y',
            'p1z',
            'p2z',
            'f1x',
            'f2x',
            'f1y',
            'f2y',
            'f1z',
            'f2z']
        for sf in signed_floats:
            fmt[sf] = self.fmt_class['signed float']
        unsigned_floats = ['d', 'df', 'af', 'p1', 'p2', 'f1', 'f2']
        for usf in unsigned_floats:
            fmt[usf] = self.fmt_class['unsigned float']
        integers = ['i', 'an', 't'] + ['r' + sf for sf in signed_floats] + \
            ['r' + usf for usf in unsigned_floats]
        for i in integers:
            fmt[i] = self.fmt_class['int']
        fmt['el'] = self.fmt_class['conv']

        self.fmt = fmt


class Table:
    def __init__(self,
                 field_specs,
                 summary_functions=[],
                 tableformat=None,
                 max_lines=None,
                 title='',
                 tablewidth=None):

        self.max_lines = max_lines
        self.summary_functions = summary_functions
        self.field_specs = field_specs

        self.fields, self.hier, self.scent = parse_field_specs(self.field_specs)
        self.nfields = len(self.fields)

        # formatting
        if tableformat is None:
            self.tableformat = TableFormat()
        else:
            self.tableformat = tableformat

        if tablewidth is None:
            self.tablewidth = self.tableformat.columnwidth * self.nfields
        else:
            self.tablewidth = tablewidth

        self.title = title

    def make(self, atoms1, atoms2, csv=False):
        header = self.make_header(csv=csv)
        body = self.make_body(atoms1, atoms2, csv=csv)
        if self.max_lines is not None:
            body = body[:self.max_lines]
        summary = self.make_summary(atoms1, atoms2)

        return '\n'.join([self.title,
                          self.tableformat.toprule * self.tablewidth,
                          header,
                          self.tableformat.midrule * self.tablewidth,
                          body,
                          self.tableformat.bottomrule * self.tablewidth,
                          summary])

    def make_header(self, csv=False):
        if csv:
            return ','.join([header_alias(field) for field in self.fields])

        fields = self.tableformat.fmt_class['str'] * self.nfields
        headers = [header_alias(field) for field in self.fields]

        return self.tableformat.formatter(fields, *headers)

    def make_summary(self, atoms1, atoms2):
        return '\n'.join([summary_function(atoms1, atoms2)
                          for summary_function in self.summary_functions])

    def make_body(self, atoms1, atoms2, csv=False):
        field_data = np.array([get_field_data(atoms1, atoms2, field)
                               for field in self.fields])

        sorting_array = field_data * self.scent[:, np.newaxis]
        sorting_array = sorting_array[self.hier]
        sorting_array = prec_round(sorting_array, self.tableformat.precision)

        field_data = field_data[:, np.lexsort(sorting_array)].transpose()

        if csv:
            rowformat = ','.join(
                ['{:h}' if field == 'el' else '{{:.{}E}}'.format(
                    self.tableformat.precision) for field in self.fields])
        else:
            rowformat = ''.join([self.tableformat.fmt[field]
                                 for field in self.fields])
        body = [
            self.tableformat.formatter(
                rowformat,
                *row) for row in field_data]
        return '\n'.join(body)


default_index = string2index(':')


def slice_split(filename):
    if '@' in filename:
        filename, index = parse_filename(filename, None)
    else:
        filename, index = parse_filename(filename, default_index)
    return filename, index
