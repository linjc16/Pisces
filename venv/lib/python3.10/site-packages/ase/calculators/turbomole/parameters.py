# fmt: off

# type: ignore
"""turbomole parameters management classes and functions"""

import os
import re
from math import floor, log10

import numpy as np

from ase.calculators.turbomole.reader import parse_data_group, read_data_group
from ase.calculators.turbomole.writer import add_data_group, delete_data_group
from ase.units import Bohr, Ha


class TurbomoleParameters(dict):
    """class to manage turbomole parameters"""

    available_functionals = [
        'slater-dirac-exchange', 's-vwn', 'vwn', 's-vwn_Gaussian', 'pwlda',
        'becke-exchange', 'b-lyp', 'b-vwn', 'lyp', 'b-p', 'pbe', 'tpss',
        'bh-lyp', 'b3-lyp', 'b3-lyp_Gaussian', 'pbe0', 'tpssh', 'lhf', 'oep',
        'b97-d', 'b2-plyp'
    ]

    # nested dictionary with parameters attributes
    parameter_spec = {
        'automatic orbital shift': {
            'comment': None,
            'default': 0.1,
            'group': 'scforbitalshift',
            'key': 'automatic',
            'mapping': {
                'to_control': lambda a: a / Ha,
                'from_control': lambda a: a * Ha
            },
            'type': float,
            'units': 'eV',
            'updateable': True
        },
        'basis set definition': {
            'comment': 'used only in restart',
            'default': None,
            'group': 'basis',
            'key': None,
            'type': list,
            'units': None,
            'updateable': False
        },
        'basis set name': {
            'comment': 'current default from module "define"',
            'default': 'def-SV(P)',
            'group': 'basis',
            'key': None,
            'type': str,
            'units': None,
            'updateable': False
        },
        'closed-shell orbital shift': {
            'comment': 'does not work with automatic',
            'default': None,
            'group': 'scforbitalshift',
            'key': 'closedshell',
            'mapping': {
                'to_control': lambda a: a / Ha,
                'from_control': lambda a: a * Ha
            },
            'type': float,
            'units': 'eV',
            'updateable': True
        },
        'damping adjustment step': {
            'comment': None,
            'default': None,
            'group': 'scfdamp',
            'key': 'step',
            'type': float,
            'units': None,
            'updateable': True
        },
        'default eht atomic orbitals': {
            'comment': None,
            'default': None,
            'group': None,
            'key': None,
            'type': bool,
            'units': None,
            'updateable': False
        },
        'density convergence': {
            'comment': None,
            'default': None,
            'group': 'denconv',
            'key': 'denconv',
            'non-define': True,
            'type': float,
            'units': None,
            'updateable': True
        },
        'density functional': {
            'comment': None,
            'default': 'b-p',
            'group': 'dft',
            'key': 'functional',
            'type': str,
            'units': None,
            'updateable': True
        },
        'energy convergence': {
            'comment': 'jobex -energy <int>',
            'default': None,
            'group': None,
            'key': None,
            'mapping': {
                'to_control': lambda a: a / Ha,
                'from_control': lambda a: a * Ha
            },
            'type': float,
            'units': 'eV',
            'updateable': True
        },
        'esp fit': {
            'comment': 'ESP fit',
            'default': None,
            'group': 'esp_fit',
            'key': 'esp_fit',
            'type': str,
            'units': None,
            'updateable': True,
            'non-define': True
        },
        'fermi annealing factor': {
            'comment': None,
            'default': 0.95,
            'group': 'fermi',
            'key': 'tmfac',
            'type': float,
            'units': None,
            'updateable': True
        },
        'fermi final temperature': {
            'comment': None,
            'default': 300.,
            'group': 'fermi',
            'key': 'tmend',
            'type': float,
            'units': 'Kelvin',
            'updateable': True
        },
        'fermi homo-lumo gap criterion': {
            'comment': None,
            'default': 0.1,
            'group': 'fermi',
            'key': 'hlcrt',
            'mapping': {
                'to_control': lambda a: a / Ha,
                'from_control': lambda a: a * Ha
            },
            'type': float,
            'units': 'eV',
            'updateable': True
        },
        'fermi initial temperature': {
            'comment': None,
            'default': 300.,
            'group': 'fermi',
            'key': 'tmstrt',
            'type': float,
            'units': 'Kelvin',
            'updateable': True
        },
        'fermi stopping criterion': {
            'comment': None,
            'default': 0.001,
            'group': 'fermi',
            'key': 'stop',
            'mapping': {
                'to_control': lambda a: a / Ha,
                'from_control': lambda a: a * Ha
            },
            'type': float,
            'units': 'eV',
            'updateable': True
        },
        'force convergence': {
            'comment': 'jobex -gcart <int>',
            'default': None,
            'group': None,
            'key': None,
            'mapping': {
                'to_control': lambda a: a / Ha * Bohr,
                'from_control': lambda a: a * Ha / Bohr
            },
            'type': float,
            'units': 'eV/Angstrom',
            'updateable': True
        },
        'geometry optimization iterations': {
            'comment': 'jobex -c <int>',
            'default': None,
            'group': None,
            'key': None,
            'type': int,
            'units': None,
            'updateable': True
        },
        'grid size': {
            'comment': None,
            'default': 'm3',
            'group': 'dft',
            'key': 'gridsize',
            'type': str,
            'units': None,
            'updateable': True
        },
        'ground state': {
            'comment': 'only this is currently supported',
            'default': True,
            'group': None,
            'key': None,
            'type': bool,
            'units': None,
            'updateable': False
        },
        'initial damping': {
            'comment': None,
            'default': None,
            'group': 'scfdamp',
            'key': 'start',
            'type': float,
            'units': None,
            'updateable': True
        },
        'initial guess': {
            'comment': '"eht", "hcore" or {"use": "<path/to/control>"}',
            'default': 'eht',
            'group': None,
            'key': None,
            'type': (str, dict),
            'units': None,
            'updateable': False
        },
        'minimal damping': {
            'comment': None,
            'default': None,
            'group': 'scfdamp',
            'key': 'min',
            'type': float,
            'units': None,
            'updateable': True
        },
        'multiplicity': {
            'comment': None,
            'default': None,
            'group': None,
            'key': None,
            'type': int,
            'units': None,
            'updateable': False
        },
        'non-automatic orbital shift': {
            'comment': None,
            'default': False,
            'group': 'scforbitalshift',
            'key': 'noautomatic',
            'type': bool,
            'units': None,
            'updateable': True
        },
        'numerical hessian': {
            'comment': 'NumForce will be used if dictionary exists',
            'default': None,
            'group': None,
            'key': None,
            'type': dict,
            'units': None,
            'updateable': True
        },
        'point group': {
            'comment': 'only c1 supported',
            'default': 'c1',
            'group': 'symmetry',
            'key': 'symmetry',
            'type': str,
            'units': None,
            'updateable': False
        },
        'ri memory': {
            'comment': None,
            'default': 1000,
            'group': 'ricore',
            'key': 'ricore',
            'type': int,
            'units': 'Megabyte',
            'updateable': True
        },
        'rohf': {
            'comment': 'used only in restart',
            'default': None,
            'group': None,
            'key': None,
            'type': bool,
            'units': None,
            'updateable': False
        },
        'scf energy convergence': {
            'comment': None,
            'default': None,
            'group': 'scfconv',
            'key': 'scfconv',
            'mapping': {
                'to_control': lambda a: int(floor(-log10(a / Ha))),
                'from_control': lambda a: 10**(-a) * Ha
            },
            'type': float,
            'units': 'eV',
            'updateable': True
        },
        'scf iterations': {
            'comment': None,
            'default': 60,
            'group': 'scfiterlimit',
            'key': 'scfiterlimit',
            'type': int,
            'units': None,
            'updateable': True
        },
        'task': {
            'comment': '"energy calculation" = "energy", '
                       '"gradient calculation" = "gradient", '
                       '"geometry optimization" = "optimize", '
                       '"normal mode analysis" = "frequencies"',
            'default': 'energy',
            'group': None,
            'key': None,
            'type': str,
            'units': None,
            'updateable': True
        },
        'title': {
            'comment': None,
            'default': '',
            'group': 'title',
            'key': 'title',
            'type': str,
            'units': None,
            'updateable': False
        },
        'total charge': {
            'comment': None,
            'default': 0,
            'group': None,
            'key': None,
            'type': int,
            'units': None,
            'updateable': False
        },
        'transition vector': {
            'comment': 'vector for transition state optimization',
            'default': None,
            'group': 'statpt',
            'key': 'itrvec',
            'type': int,
            'units': None,
            'updateable': True,
            'non-define': True
        },
        'uhf': {
            'comment': None,
            'default': None,
            'group': 'uhf',
            'key': 'uhf',
            'type': bool,
            'units': None,
            'updateable': False
        },
        'use basis set library': {
            'comment': 'only true implemented',
            'default': True,
            'group': 'basis',
            'key': None,
            'type': bool,
            'units': None,
            'updateable': False
        },
        'use dft': {
            'comment': None,
            'default': True,
            'group': 'dft',
            'key': 'dft',
            'type': bool,
            'units': None,
            'updateable': False
        },
        'use fermi smearing': {
            'comment': None,
            'default': False,
            'group': 'fermi',
            'key': 'fermi',
            'type': bool,
            'units': None,
            'updateable': True
        },
        'use redundant internals': {
            'comment': None,
            'default': False,
            'group': 'redundant',
            'key': None,
            'type': bool,
            'units': None,
            'updateable': False
        },
        'use resolution of identity': {
            'comment': None,
            'default': False,
            'group': 'rij',
            'key': 'rij',
            'type': bool,
            'units': None,
            'updateable': False
        }
    }

    spec_names = {
        'default': 'default_parameters',
        'comment': 'parameter_comment',
        'updateable': 'parameter_updateable',
        'type': 'parameter_type',
        'key': 'parameter_key',
        'group': 'parameter_group',
        'units': 'parameter_units',
        'mapping': 'parameter_mapping',
        'non-define': 'parameter_no_define'
    }
    # flat dictionaries with parameters attributes
    default_parameters = {}
    parameter_group = {}
    parameter_type = {}
    parameter_key = {}
    parameter_units = {}
    parameter_comment = {}
    parameter_updateable = {}
    parameter_mapping = {}
    parameter_no_define = {}

    def __init__(self, **kwargs):
        # construct flat dictionaries with parameter attributes
        for p in self.parameter_spec:
            for k in self.spec_names:
                if k in list(self.parameter_spec[p].keys()):
                    subdict = getattr(self, self.spec_names[k])
                    subdict.update({p: self.parameter_spec[p][k]})
        super().__init__(**self.default_parameters)
        self.update(kwargs)

    def update(self, dct):
        """check the type of parameters in dct and then update"""
        for par in dct.keys():
            if par not in self.parameter_spec:
                raise ValueError('invalid parameter: ' + par)

        for key, val in dct.items():
            correct_type = self.parameter_spec[key]['type']
            if not isinstance(val, (correct_type, type(None))):
                msg = str(key) + ' has wrong type: ' + str(type(val))
                raise TypeError(msg)
            self[key] = val

    def update_data_groups(self, params_update):
        """updates data groups in the control file"""
        # construct a list of data groups to update
        grps = []
        for p in list(params_update.keys()):
            if self.parameter_group[p] is not None:
                grps.append(self.parameter_group[p])

        # construct a dictionary of data groups and update params
        dgs = {}
        for g in grps:
            dgs[g] = {}
            for p in self.parameter_key:
                if g == self.parameter_group[p]:
                    if self.parameter_group[p] == self.parameter_key[p]:
                        if p in list(params_update.keys()):
                            val = params_update[p]
                            pmap = list(self.parameter_mapping.keys())
                            if val is not None and p in pmap:
                                fun = self.parameter_mapping[p]['to_control']
                                val = fun(params_update[p])
                            dgs[g] = val
                    else:
                        if p in list(self.params_old.keys()):
                            val = self.params_old[p]
                            pmap = list(self.parameter_mapping.keys())
                            if val is not None and p in pmap:
                                fun = self.parameter_mapping[p]['to_control']
                                val = fun(self.params_old[p])
                            dgs[g][self.parameter_key[p]] = val
                        if p in list(params_update.keys()):
                            val = params_update[p]
                            pmap = list(self.parameter_mapping.keys())
                            if val is not None and p in pmap:
                                fun = self.parameter_mapping[p]['to_control']
                                val = fun(params_update[p])
                            dgs[g][self.parameter_key[p]] = val

        # write dgs dictionary to a data group
        for g in dgs:
            delete_data_group(g)
            if isinstance(dgs[g], dict):
                string = ''
                for key in list(dgs[g].keys()):
                    if dgs[g][key] is None:
                        continue
                    elif isinstance(dgs[g][key], bool):
                        if dgs[g][key]:
                            string += ' ' + key
                    else:
                        string += ' ' + key + '=' + str(dgs[g][key])
                add_data_group(g, string=string)
            else:
                if isinstance(dgs[g], bool):
                    if dgs[g]:
                        add_data_group(g, string='')
                else:
                    add_data_group(g, string=str(dgs[g]))

    def update_no_define_parameters(self):
        """process key parameters that are not written with define"""
        for p, v in self.items():
            if self.parameter_no_define.get(p):
                if v:
                    if p in self.parameter_mapping:
                        fun = self.parameter_mapping[p]['to_control']
                        val = fun(v)
                    else:
                        val = v
                    delete_data_group(self.parameter_group[p])
                    if self.parameter_group[p] != self.parameter_key[p]:
                        val = '\n ' + self.parameter_key[p] + ' ' + str(val)
                    add_data_group(self.parameter_group[p], str(val))
                else:
                    delete_data_group(self.parameter_group[p])

    def verify(self):
        """detect wrong or not implemented parameters"""

        if getattr(self, 'define_str', None) is not None:
            assert isinstance(self.define_str, str), 'define_str must be str'
            assert len(self.define_str) != 0, 'define_str may not be empty'
        else:
            for par in self:
                assert par in self.parameter_spec, 'invalid parameter: ' + par

            if self.get('use dft'):
                func_list = [x.lower() for x in self.available_functionals]
                func = self['density functional']
                assert func.lower() in func_list, (
                    'density functional not available / not supported'
                )

            assert self['multiplicity'] is not None, 'multiplicity not defined'
            assert self['multiplicity'] > 0, 'multiplicity has wrong value'

            if self.get('rohf'):
                raise NotImplementedError('ROHF not implemented')
            if self['initial guess'] not in ['eht', 'hcore']:
                if not (isinstance(self['initial guess'], dict) and
                        'use' in self['initial guess'].keys()):
                    raise ValueError('Wrong input for initial guess')
            if not self['use basis set library']:
                raise NotImplementedError('Explicit basis set definition')
            if self['point group'] != 'c1':
                raise NotImplementedError('Point group not impemeneted')

    def get_define_str(self, natoms):
        """construct a define string from the parameters dictionary"""

        if getattr(self, 'define_str', None):
            return self.define_str

        define_str_tpl = (
            '\n__title__\na coord\n__inter__\n'
            'bb all __basis_set__\n*\neht\n__eht_aos_str__y\n__charge_str__'
            '__occ_str____single_atom_str____norb_str____dft_str____ri_str__'
            '__scfiterlimit____fermi_str____damp_str__q\n'
        )

        params = self

        if params['use redundant internals']:
            internals_str = 'ired\n*'
        else:
            internals_str = '*\nno'
        charge_str = str(params['total charge']) + '\n'

        if params['multiplicity'] == 1:
            if params['uhf']:
                occ_str = 'n\ns\n*\n'
            else:
                occ_str = 'y\n'
        elif params['multiplicity'] == 2:
            occ_str = 'y\n'
        elif params['multiplicity'] == 3:
            occ_str = 'n\nt\n*\n'
        else:
            unpaired = params['multiplicity'] - 1
            if params['use fermi smearing']:
                occ_str = 'n\nuf ' + str(unpaired) + '\n*\n'
            else:
                occ_str = 'n\nu ' + str(unpaired) + '\n*\n'

        if natoms != 1:
            single_atom_str = ''
        else:
            single_atom_str = '\n'

        if params['multiplicity'] == 1 and not params['uhf']:
            norb_str = ''
        else:
            norb_str = 'n\n'

        if params['use dft']:
            dft_str = 'dft\non\n*\n'
        else:
            dft_str = ''

        if params['density functional']:
            dft_str += 'dft\nfunc ' + params['density functional'] + '\n*\n'

        if params['grid size']:
            dft_str += 'dft\ngrid ' + params['grid size'] + '\n*\n'

        if params['use resolution of identity']:
            ri_str = 'ri\non\nm ' + str(params['ri memory']) + '\n*\n'
        else:
            ri_str = ''

        if params['scf iterations']:
            scfmaxiter = params['scf iterations']
            scfiter_str = 'scf\niter\n' + str(scfmaxiter) + '\n\n'
        else:
            scfiter_str = ''
        if params['scf energy convergence']:
            conv = floor(-log10(params['scf energy convergence'] / Ha))
            scfiter_str += 'scf\nconv\n' + str(int(conv)) + '\n\n'

        fermi_str = ''
        if params['use fermi smearing']:
            fermi_str = 'scf\nfermi\n'
            if params['fermi initial temperature']:
                par = str(params['fermi initial temperature'])
                fermi_str += '1\n' + par + '\n'
            if params['fermi final temperature']:
                par = str(params['fermi final temperature'])
                fermi_str += '2\n' + par + '\n'
            if params['fermi annealing factor']:
                par = str(params['fermi annealing factor'])
                fermi_str += '3\n' + par + '\n'
            if params['fermi homo-lumo gap criterion']:
                par = str(params['fermi homo-lumo gap criterion'])
                fermi_str += '4\n' + par + '\n'
            if params['fermi stopping criterion']:
                par = str(params['fermi stopping criterion'])
                fermi_str += '5\n' + par + '\n'
            fermi_str += '\n\n'

        damp_str = ''
        damp_keys = ('initial damping', 'damping adjustment step',
                     'minimal damping')
        damp_pars = [params[k] for k in damp_keys]
        if any(damp_pars):
            damp_str = 'scf\ndamp\n'
            for par in damp_pars:
                par_str = str(par) if par else ''
                damp_str += par_str + '\n'
            damp_str += '\n'

        eht_aos_str = 'y\n' if params['default eht atomic orbitals'] else ''

        define_str = define_str_tpl
        define_str = re.sub('__title__', params['title'], define_str)
        define_str = re.sub('__basis_set__', params['basis set name'],
                            define_str)
        define_str = re.sub('__charge_str__', charge_str, define_str)
        define_str = re.sub('__occ_str__', occ_str, define_str)
        define_str = re.sub('__norb_str__', norb_str, define_str)
        define_str = re.sub('__dft_str__', dft_str, define_str)
        define_str = re.sub('__ri_str__', ri_str, define_str)
        define_str = re.sub('__single_atom_str__', single_atom_str,
                            define_str)
        define_str = re.sub('__inter__', internals_str, define_str)
        define_str = re.sub('__scfiterlimit__', scfiter_str, define_str)
        define_str = re.sub('__fermi_str__', fermi_str, define_str)
        define_str = re.sub('__damp_str__', damp_str, define_str)
        define_str = re.sub('__eht_aos_str__', eht_aos_str, define_str)

        return define_str

    def read_restart(self, atoms, results):
        """read parameters from control file"""

        params = {}
        pdgs = {
            p: parse_data_group(
                read_data_group(
                    self.parameter_group[p]), self.parameter_group[p]
            )
            for p in self.parameter_group
            if self.parameter_group[p] and self.parameter_key[p]
        }
        for p in self.parameter_key:
            if self.parameter_key[p]:
                if self.parameter_key[p] == self.parameter_group[p]:
                    if pdgs[p] is None:
                        if self.parameter_type[p] is bool:
                            params[p] = False
                        else:
                            params[p] = None
                    else:
                        if self.parameter_type[p] is bool:
                            params[p] = True
                        else:
                            typ = self.parameter_type[p]
                            val = typ(pdgs[p])
                            mapping = self.parameter_mapping
                            if p in list(mapping.keys()):
                                fun = mapping[p]['from_control']
                                val = fun(val)
                            params[p] = val
                else:
                    if pdgs[p] is None:
                        params[p] = None
                    elif isinstance(pdgs[p], str):
                        if self.parameter_type[p] is bool:
                            params[p] = (pdgs[p] == self.parameter_key[p])
                    else:
                        if self.parameter_key[p] not in list(pdgs[p].keys()):
                            if self.parameter_type[p] is bool:
                                params[p] = False
                            else:
                                params[p] = None
                        else:
                            typ = self.parameter_type[p]
                            val = typ(pdgs[p][self.parameter_key[p]])
                            mapping = self.parameter_mapping
                            if p in list(mapping.keys()):
                                fun = mapping[p]['from_control']
                                val = fun(val)
                            params[p] = val

        # non-group or non-key parameters

        # per-element and per-atom basis sets not implemented in calculator
        basis_sets = {bs['nickname'] for bs in results['basis set']}
        assert len(basis_sets) == 1
        params['basis set name'] = list(basis_sets)[0]
        params['basis set definition'] = results['basis set']

        # rohf, multiplicity and total charge
        orbs = results['molecular orbitals']
        params['rohf'] = (bool(len(read_data_group('rohf'))) or
                          bool(len(read_data_group('roothaan'))))
        core_charge = 0
        if results['ecps']:
            for ecp in results['ecps']:
                for symbol in atoms.get_chemical_symbols():
                    if symbol.lower() == ecp['element'].lower():
                        core_charge -= ecp['number of core electrons']
        if params['uhf']:
            alpha_occ = [o['occupancy'] for o in orbs if o['spin'] == 'alpha']
            beta_occ = [o['occupancy'] for o in orbs if o['spin'] == 'beta']
            spin = (np.sum(alpha_occ) - np.sum(beta_occ)) * 0.5
            params['multiplicity'] = int(2 * spin + 1)
            nuclear_charge = int(sum(atoms.numbers))
            electron_charge = -int(sum(alpha_occ) + sum(beta_occ))
            electron_charge += core_charge
            params['total charge'] = nuclear_charge + electron_charge
        elif not params['rohf']:  # restricted HF (closed shell)
            params['multiplicity'] = 1
            nuclear_charge = int(sum(atoms.numbers))
            electron_charge = -int(sum(o['occupancy'] for o in orbs))
            electron_charge += core_charge
            params['total charge'] = nuclear_charge + electron_charge
        else:
            raise NotImplementedError('ROHF not implemented')

        # task-related parameters
        if os.path.exists('job.start'):
            with open('job.start') as log:
                lines = log.readlines()
            for line in lines:
                if 'CRITERION FOR TOTAL SCF-ENERGY' in line:
                    en = int(re.search(r'10\*{2}\((-\d+)\)', line).group(1))
                    mapp = self.parameter_mapping['energy convergence']
                    params['energy convergence'] = mapp['from_control'](10**en)
                if 'CRITERION FOR MAXIMUM NORM OF SCF-ENERGY GRADIENT' in line:
                    gr = int(re.search(r'10\*{2}\((-\d+)\)', line).group(1))
                    mapp = self.parameter_mapping['force convergence']
                    params['force convergence'] = mapp['from_control'](10**gr)
                if 'AN OPTIMIZATION WITH MAX' in line:
                    cy = int(re.search(r'MAX. (\d+) CYCLES', line).group(1))
                    params['geometry optimization iterations'] = cy
        self.update(params)
        self.params_old = params

    def update_restart(self, dct):
        """update parameters after a restart"""
        nulst = [k for k in dct.keys() if not self.parameter_updateable[k]]
        if len(nulst) != 0:
            raise ValueError(f'parameters {nulst} cannot be changed')
        self.update(dct)
        self.update_data_groups(dct)
