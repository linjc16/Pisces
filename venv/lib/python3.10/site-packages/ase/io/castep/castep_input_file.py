# fmt: off

import difflib
import re
import warnings
from typing import List, Set

import numpy as np

from ase import Atoms

# A convenient table to avoid the previously used "eval"
_tf_table = {
    '': True,  # Just the keyword is equivalent to True
    'True': True,
    'False': False}


def _parse_tss_block(value, scaled=False):
    # Parse the assigned value for a Transition State Search structure block
    is_atoms = isinstance(value, Atoms)
    try:
        is_strlist = all(map(lambda x: isinstance(x, str), value))
    except TypeError:
        is_strlist = False

    if not is_atoms:
        if not is_strlist:
            # Invalid!
            raise TypeError('castep.cell.positions_abs/frac_intermediate/'
                            'product expects Atoms object or list of strings')

        # First line must be Angstroms, or nothing
        has_units = len(value[0].strip().split()) == 1
        if (not scaled) and has_units and value[0].strip() != 'ang':
            raise RuntimeError('Only ang units currently supported in castep.'
                               'cell.positions_abs_intermediate/product')
        return '\n'.join(map(str.strip, value))
    else:
        text_block = '' if scaled else 'ang\n'
        positions = (value.get_scaled_positions() if scaled else
                     value.get_positions())
        symbols = value.get_chemical_symbols()
        for s, p in zip(symbols, positions):
            text_block += '    {} {:.3f} {:.3f} {:.3f}\n'.format(s, *p)

        return text_block


class CastepOption:
    """"A CASTEP option. It handles basic conversions from string to its value
    type."""

    default_convert_types = {
        'boolean (logical)': 'bool',
        'defined': 'bool',
        'string': 'str',
        'integer': 'int',
        'real': 'float',
        'integer vector': 'int_vector',
        'real vector': 'float_vector',
        'physical': 'float_physical',
        'block': 'block'
    }

    def __init__(self, keyword, level, option_type, value=None,
                 docstring='No information available'):
        self.keyword = keyword
        self.level = level
        self.type = option_type
        self._value = value
        self.__doc__ = docstring

    @property
    def value(self):

        if self._value is not None:
            if self.type.lower() in ('integer vector', 'real vector',
                                     'physical'):
                return ' '.join(map(str, self._value))
            elif self.type.lower() in ('boolean (logical)', 'defined'):
                return str(self._value).upper()
            else:
                return str(self._value)

    @property
    def raw_value(self):
        # The value, not converted to a string
        return self._value

    @value.setter  # type: ignore[attr-defined, no-redef]
    def value(self, val):

        if val is None:
            self.clear()
            return

        ctype = self.default_convert_types.get(self.type.lower(), 'str')
        typeparse = f'_parse_{ctype}'
        try:
            self._value = getattr(self, typeparse)(val)
        except ValueError:
            raise ConversionError(ctype, self.keyword, val)

    def clear(self):
        """Reset the value of the option to None again"""
        self._value = None

    @staticmethod
    def _parse_bool(value):
        try:
            value = _tf_table[str(value).strip().title()]
        except (KeyError, ValueError):
            raise ValueError()
        return value

    @staticmethod
    def _parse_str(value):
        value = str(value)
        return value

    @staticmethod
    def _parse_int(value):
        value = int(value)
        return value

    @staticmethod
    def _parse_float(value):
        value = float(value)
        return value

    @staticmethod
    def _parse_int_vector(value):
        # Accepts either a string or an actual list/numpy array of ints
        if isinstance(value, str):
            if ',' in value:
                value = value.replace(',', ' ')
            value = list(map(int, value.split()))

        value = np.array(value)

        if value.shape != (3,) or value.dtype != int:
            raise ValueError()

        return list(value)

    @staticmethod
    def _parse_float_vector(value):
        # Accepts either a string or an actual list/numpy array of floats
        if isinstance(value, str):
            if ',' in value:
                value = value.replace(',', ' ')
            value = list(map(float, value.split()))

        value = np.array(value) * 1.0

        if value.shape != (3,) or value.dtype != float:
            raise ValueError()

        return list(value)

    @staticmethod
    def _parse_float_physical(value):
        # If this is a string containing units, saves them
        if isinstance(value, str):
            value = value.split()

        try:
            l = len(value)
        except TypeError:
            l = 1
            value = [value]

        if l == 1:
            try:
                value = (float(value[0]), '')
            except (TypeError, ValueError):
                raise ValueError()
        elif l == 2:
            try:
                value = (float(value[0]), value[1])
            except (TypeError, ValueError, IndexError):
                raise ValueError()
        else:
            raise ValueError()

        return value

    @staticmethod
    def _parse_block(value):

        if isinstance(value, str):
            return value
        elif hasattr(value, '__getitem__'):
            return '\n'.join(value)  # Arrays of lines
        else:
            raise ValueError()

    def __repr__(self):
        if self._value:
            expr = ('Option: {keyword}({type}, {level}):\n{_value}\n'
                    ).format(**self.__dict__)
        else:
            expr = ('Option: {keyword}[unset]({type}, {level})'
                    ).format(**self.__dict__)
        return expr

    def __eq__(self, other):
        if not isinstance(other, CastepOption):
            return False
        else:
            return self.__dict__ == other.__dict__


class CastepOptionDict:
    """A dictionary-like object to hold a set of options for .cell or .param
    files loaded from a dictionary, for the sake of validation.

    Replaces the old CastepCellDict and CastepParamDict that were defined in
    the castep_keywords.py file.
    """

    def __init__(self, options=None):
        object.__init__(self)
        self._options = {}  # ComparableDict is not needed any more as
        # CastepOptions can be compared directly now
        for kw in options:
            opt = CastepOption(**options[kw])
            self._options[opt.keyword] = opt
            self.__dict__[opt.keyword] = opt


class CastepInputFile:

    """Master class for CastepParam and CastepCell to inherit from"""

    _keyword_conflicts: List[Set[str]] = []

    def __init__(self, options_dict=None, keyword_tolerance=1):
        object.__init__(self)

        if options_dict is None:
            options_dict = CastepOptionDict({})

        self._options = options_dict._options
        self.__dict__.update(self._options)
        # keyword_tolerance means how strict the checks on new attributes are
        # 0 = no new attributes allowed
        # 1 = new attributes allowed, warning given
        # 2 = new attributes allowed, silent
        self._perm = np.clip(keyword_tolerance, 0, 2)

        # Compile a dictionary for quick check of conflict sets
        self._conflict_dict = {
            kw: set(cset).difference({kw})
            for cset in self._keyword_conflicts for kw in cset}

    def __repr__(self):
        expr = ''
        is_default = True
        for key, option in sorted(self._options.items()):
            if option.value is not None:
                is_default = False
                expr += ('%20s : %s\n' % (key, option.value))
        if is_default:
            expr = 'Default\n'

        expr += f'Keyword tolerance: {self._perm}'
        return expr

    def __setattr__(self, attr, value):

        # Hidden attributes are treated normally
        if attr.startswith('_'):
            self.__dict__[attr] = value
            return

        if attr not in self._options.keys():

            if self._perm > 0:
                # Do we consider it a string or a block?
                is_str = isinstance(value, str)
                is_block = False
                if ((hasattr(value, '__getitem__') and not is_str)
                        or (is_str and len(value.split('\n')) > 1)):
                    is_block = True

            if self._perm == 0:
                similars = difflib.get_close_matches(attr,
                                                     self._options.keys())
                if similars:
                    raise RuntimeError(
                        f'Option "{attr}" not known! You mean "{similars[0]}"?')
                else:
                    raise RuntimeError(f'Option "{attr}" is not known!')
            elif self._perm == 1:
                warnings.warn(('Option "%s" is not known and will '
                               'be added as a %s') % (attr,
                                                      ('block' if is_block else
                                                       'string')))
            attr = attr.lower()
            opt = CastepOption(keyword=attr, level='Unknown',
                               option_type='block' if is_block else 'string')
            self._options[attr] = opt
            self.__dict__[attr] = opt
        else:
            attr = attr.lower()
            opt = self._options[attr]

        if not opt.type.lower() == 'block' and isinstance(value, str):
            value = value.replace(':', ' ')

        # If it is, use the appropriate parser, unless a custom one is defined
        attrparse = f'_parse_{attr.lower()}'

        # Check for any conflicts if the value is not None
        if value is not None:
            cset = self._conflict_dict.get(attr.lower(), {})
            for c in cset:
                if (c in self._options and self._options[c].value):
                    warnings.warn(
                        'option "{attr}" conflicts with "{conflict}" in '
                        'calculator. Setting "{conflict}" to '
                        'None.'.format(attr=attr, conflict=c))
                    self._options[c].value = None

        if hasattr(self, attrparse):
            self._options[attr].value = self.__getattribute__(attrparse)(value)
        else:
            self._options[attr].value = value

    def __getattr__(self, name):
        if name[0] == '_' or self._perm == 0:
            raise AttributeError()

        if self._perm == 1:
            warnings.warn(f'Option {(name)} is not known, returning None')

        return CastepOption(keyword='none', level='Unknown',
                            option_type='string', value=None)

    def get_attr_dict(self, raw=False, types=False):
        """Settings that go into .param file in a traditional dict"""

        attrdict = {k: o.raw_value if raw else o.value
                    for k, o in self._options.items() if o.value is not None}

        if types:
            for key, val in attrdict.items():
                attrdict[key] = (val, self._options[key].type)

        return attrdict


class CastepParam(CastepInputFile):
    """CastepParam abstracts the settings that go into the .param file"""

    _keyword_conflicts = [{'cut_off_energy', 'basis_precision'}, ]

    def __init__(self, castep_keywords, keyword_tolerance=1):
        self._castep_version = castep_keywords.castep_version
        CastepInputFile.__init__(self, castep_keywords.CastepParamDict(),
                                 keyword_tolerance)

    @property
    def castep_version(self):
        return self._castep_version

    # .param specific parsers
    def _parse_reuse(self, value):
        if value is None:
            return None  # Reset the value
        try:
            if self._options['continuation'].value:
                warnings.warn('Cannot set reuse if continuation is set, and '
                              'vice versa. Set the other to None, if you want '
                              'this setting.')
                return None
        except KeyError:
            pass
        return 'default' if (value is True) else str(value)

    def _parse_continuation(self, value):
        if value is None:
            return None  # Reset the value
        try:
            if self._options['reuse'].value:
                warnings.warn('Cannot set reuse if continuation is set, and '
                              'vice versa. Set the other to None, if you want '
                              'this setting.')
                return None
        except KeyError:
            pass
        return 'default' if (value is True) else str(value)


class CastepCell(CastepInputFile):

    """CastepCell abstracts all setting that go into the .cell file"""

    _keyword_conflicts = [
        {'kpoint_mp_grid', 'kpoint_mp_spacing', 'kpoint_list',
         'kpoints_mp_grid', 'kpoints_mp_spacing', 'kpoints_list'},
        {'bs_kpoint_mp_grid',
         'bs_kpoint_mp_spacing',
         'bs_kpoint_list',
         'bs_kpoint_path',
         'bs_kpoints_mp_grid',
         'bs_kpoints_mp_spacing',
         'bs_kpoints_list',
         'bs_kpoints_path'},
        {'spectral_kpoint_mp_grid',
         'spectral_kpoint_mp_spacing',
         'spectral_kpoint_list',
         'spectral_kpoint_path',
         'spectral_kpoints_mp_grid',
         'spectral_kpoints_mp_spacing',
         'spectral_kpoints_list',
         'spectral_kpoints_path'},
        {'phonon_kpoint_mp_grid',
         'phonon_kpoint_mp_spacing',
         'phonon_kpoint_list',
         'phonon_kpoint_path',
         'phonon_kpoints_mp_grid',
         'phonon_kpoints_mp_spacing',
         'phonon_kpoints_list',
         'phonon_kpoints_path'},
        {'fine_phonon_kpoint_mp_grid',
         'fine_phonon_kpoint_mp_spacing',
         'fine_phonon_kpoint_list',
         'fine_phonon_kpoint_path'},
        {'magres_kpoint_mp_grid',
         'magres_kpoint_mp_spacing',
         'magres_kpoint_list',
         'magres_kpoint_path'},
        {'elnes_kpoint_mp_grid',
         'elnes_kpoint_mp_spacing',
         'elnes_kpoint_list',
         'elnes_kpoint_path'},
        {'optics_kpoint_mp_grid',
         'optics_kpoint_mp_spacing',
         'optics_kpoint_list',
         'optics_kpoint_path'},
        {'supercell_kpoint_mp_grid',
         'supercell_kpoint_mp_spacing',
         'supercell_kpoint_list',
         'supercell_kpoint_path'}, ]

    def __init__(self, castep_keywords, keyword_tolerance=1):
        self._castep_version = castep_keywords.castep_version
        CastepInputFile.__init__(self, castep_keywords.CastepCellDict(),
                                 keyword_tolerance)

    @property
    def castep_version(self):
        return self._castep_version

    # .cell specific parsers
    def _parse_species_pot(self, value):

        # Single tuple
        if isinstance(value, tuple) and len(value) == 2:
            value = [value]
        # List of tuples
        if hasattr(value, '__getitem__'):
            pspots = [tuple(map(str.strip, x)) for x in value]
            if not all(map(lambda x: len(x) == 2, value)):
                warnings.warn(
                    'Please specify pseudopotentials in python as '
                    'a tuple or a list of tuples formatted like: '
                    '(species, file), e.g. ("O", "path-to/O_OTFG.usp") '
                    'Anything else will be ignored')
                return None

        text_block = self._options['species_pot'].value

        text_block = text_block if text_block else ''
        # Remove any duplicates
        for pp in pspots:
            text_block = re.sub(fr'\n?\s*{pp[0]}\s+.*', '', text_block)
            if pp[1]:
                text_block += '\n%s %s' % pp

        return text_block

    def _parse_symmetry_ops(self, value):
        if not isinstance(value, tuple) \
           or not len(value) == 2 \
           or not value[0].shape[1:] == (3, 3) \
           or not value[1].shape[1:] == (3,) \
           or not value[0].shape[0] == value[1].shape[0]:
            warnings.warn('Invalid symmetry_ops block, skipping')
            return
        # Now on to print...
        text_block = ''
        for op_i, (op_rot, op_tranls) in enumerate(zip(*value)):
            text_block += '\n'.join([' '.join([str(x) for x in row])
                                     for row in op_rot])
            text_block += '\n'
            text_block += ' '.join([str(x) for x in op_tranls])
            text_block += '\n\n'

        return text_block

    def _parse_positions_abs_intermediate(self, value):
        return _parse_tss_block(value)

    def _parse_positions_abs_product(self, value):
        return _parse_tss_block(value)

    def _parse_positions_frac_intermediate(self, value):
        return _parse_tss_block(value, True)

    def _parse_positions_frac_product(self, value):
        return _parse_tss_block(value, True)


class ConversionError(Exception):

    """Print customized error for options that are not converted correctly
    and point out that they are maybe not implemented, yet"""

    def __init__(self, key_type, attr, value):
        Exception.__init__(self)
        self.key_type = key_type
        self.value = value
        self.attr = attr

    def __str__(self):
        contact_email = 'simon.rittmeyer@tum.de'
        return f'Could not convert {self.attr} = {self.value} '\
            + 'to {self.key_type}\n' \
            + 'This means you either tried to set a value of the wrong\n'\
            + 'type or this keyword needs some special care. Please feel\n'\
            + 'to add it to the corresponding __setattr__ method and send\n'\
            + f'the patch to {(contact_email)}, so we can all benefit.'
