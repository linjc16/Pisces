import json
import urllib.request
import warnings
from collections import namedtuple
from io import StringIO
from urllib.error import HTTPError, URLError

from ase.io import read

base_url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'

PubchemSearch = namedtuple('PubchemSearch', 'search field')


class PubchemData:
    """
    a specialized class for entries from the pubchem database
    """

    def __init__(self, atoms, data):
        self.atoms = atoms
        self.data = data

    def get_atoms(self):
        return self.atoms

    def get_pubchem_data(self):
        return self.data


def search_pubchem_raw(search, field, silent=False):
    """
    A helper function for searching pubchem.

    Parameters:
        search (str or int):
            the compound you are searching for. This can be either
            a common name, CID, or smiles string depending of the
            `field` you are searching

        field (str):
            the particular field you are searching with. Possible values
            are 'name', 'CID', and 'smiles'.'name' will search common '
            'names,CID will search the Pubchem Chemical Idenitification '
            'Numberswhich can be found on their website and smiles'
            ' searches for compounds with the entered smiles string.

    returns:
        data (str):
            a string containing the raw response from pubchem.
    """
    suffix = 'sdf?record_type=3d'

    url = (
        f'{base_url}/{field}/{search!s}/{suffix}'
        if field == 'conformers'
        else f'{base_url}/compound/{field}/{search!s}/{suffix}'
    )
    try:
        r = urllib.request.urlopen(url)
    except HTTPError as e:
        raise ValueError(
            f'the search term {search} could not be found for the field {field}'
        ) from e
    except URLError as e:
        raise ValueError(
            "Couldn't reach the pubchem servers, check your internet connection"
        ) from e

    # check if there are confomers and warn them if there are
    if field != 'conformers' and not silent:
        conformer_ids = available_conformer_search(search, field)
        if len(conformer_ids) > 1:
            warnings.warn(
                f'The structure "{search}" has more than one conformer in '
                'PubChem. By default, the first conformer is returned, please '
                'ensure you are using the structure you intend to or use the '
                '`ase.data.pubchem.pubchem_conformer_search` function'
            )

    return r.read().decode('utf-8')


def parse_pubchem_raw(data):
    """
    a helper function for parsing the returned pubchem entries

    Parameters:
        data (str):
            the raw output from pubchem in string form

    returns:
        atoms (ASE Atoms Object):
            An ASE atoms obejct containing the information from
            pubchem
        pubchem_data (dict):
            a dictionary containing the non-structural information
            from pubchem

    """
    if 'PUBCHEM_COMPOUND_CID' not in data:
        raise Exception('There was a problem with the data returned by PubChem')
    f_like = StringIO(data)
    atoms = read(f_like, format='sdf')

    # check if there are confomers and warn them if there are

    # further analyze the text returned from pubchem
    pubchem_data = {}
    other_info = data.split('END\n')[1]
    other_info = other_info.split('$')[0]  # remove the $$$$ at the end
    # the strucuture of this string is > <field>\nentry_info\n
    other_info = other_info.split('> <')  # split into the fields
    for data_field in other_info:
        if data_field == '':
            continue
        field_name, entry_value = data_field.split('>\n')
        # split it into lines and remove the empty lines
        entry_value = entry_value.splitlines()
        entry_value = [a for a in entry_value if a != '']
        if len(entry_value) == 1:
            entry_value = entry_value[0]
        pubchem_data[field_name] = entry_value
    # recover partial charges
    if 'PUBCHEM_MMFF94_PARTIAL_CHARGES' in pubchem_data:
        # the first entry just contains the number of atoms with charges
        charges = pubchem_data['PUBCHEM_MMFF94_PARTIAL_CHARGES'][1:]
        # each subsequent entry contains the index and charge of the atoms
        atom_charges = [0.0] * len(atoms)
        for charge in charges:
            i, charge = charge.split()
            # indices start at 1
            atom_charges[int(i) - 1] = float(charge)
        atoms.set_initial_charges(atom_charges)
    return atoms, pubchem_data


def analyze_input(
    name=None, cid=None, smiles=None, conformer=None, silent=False
):
    """
    helper function to translate keyword arguments from intialization
    and searching into the search and field that is being asked for

    Parameters:
        see `ase.data.pubchem.pubchem_search`
    returns:
        search:
            the search term the user has entered
        field:
            the name of the field being asked for

    """
    inputs = [name, cid, smiles, conformer]
    inputs_check = [a is not None for a in [name, cid, smiles, conformer]]
    input_fields = ['name', 'cid', 'smiles', 'conformers']

    if inputs_check.count(True) > 1:
        raise ValueError(
            'Only one search term my be entered a time.'
            ' Please pass in only one of the following: '
            'name, cid, smiles, confomer'
        )
    if inputs_check.count(True) == 0:
        raise ValueError(
            'No search was entered.'
            ' Please pass in only one of the following: '
            'name, cid, smiles, confomer'
        )

    # Figure out which input has been passed in
    index = inputs_check.index(True)
    field = input_fields[index]
    search = inputs[index]

    # convert hash (triple bond) to hex for URL
    if isinstance(search, str):
        search = search.replace('#', '%23')

    return PubchemSearch(search, field)


def available_conformer_search(search, field) -> list:
    """
    Helper function to get the conformer IDs. This searches pubchem for
    the conformers of a given structure and returns all the confomer ids
    of a structure.

    Parameters:
        search (str or int):
            the compound you are searching for. This can be either
            a common name, CID, or smiles string depending of the
            `field` you are searching

        field (str):
            the particular field you are searching with. Possible values
            are 'name', 'CID', and 'smiles'.'name' will search common '
            'names,CID will search the Pubchem Chemical Idenitification '
            'Numberswhich can be found on their website and smiles'
            ' searches for compounds with the entered smiles string.

        returns:
            conformers_ids (list):
                a list of the conformer IDs from PubChem, this is different
                than the CID numbers
    """
    suffix = 'conformers/JSON'
    url = f'{base_url}/compound/{field}/{search!s}/{suffix}'
    try:
        r = urllib.request.urlopen(url)
    except HTTPError as e:
        err = ValueError(
            f'the search term {search} could not be found for the field {field}'
        )
        raise err from e
    except URLError as e:
        err = ValueError(
            "Couldn't reach the pubchem servers, check your internet connection"
        )
        raise err from e
    record = r.read().decode('utf-8')
    record = json.loads(record)
    return record['InformationList']['Information'][0]['ConformerID']


def pubchem_search(*args, **kwargs) -> PubchemData:
    """
    Search PubChem for the field and search input on the argument passed in
    returning a PubchemData object. Note that only one argument may be passed
    in at a time.

    Parameters:
        name (str):
            the common name of the compound you're searching for
        cid (str or int):
            the cid of the compound you're searching for
        smiles (str):
            the smiles string of the compound you're searching for
        conformer (str or int):
            the conformer id of the compound you're searching for

    returns:
        result (PubchemData):
            a pubchem data object containing the information on the
            requested entry
    """
    search, field = analyze_input(*args, **kwargs)
    raw_pubchem = search_pubchem_raw(search, field)
    atoms, data = parse_pubchem_raw(raw_pubchem)
    return PubchemData(atoms, data)


def pubchem_conformer_search(*args, **kwargs) -> list:
    """
    Search PubChem for all the conformers of a given compound.
    Note that only one argument may be passed in at a time.

    Parameters:
        see `ase.data.pubchem.pubchem_search`

    returns:
        conformers (list):
            a list containing the PubchemData objects of all the conformers
            for your search
    """
    search, field = analyze_input(*args, **kwargs)
    conformer_ids = available_conformer_search(search, field)
    return [pubchem_search(conformer=id_) for id_ in conformer_ids]


def pubchem_atoms_search(*args, **kwargs):
    """
    Search PubChem for the field and search input on the argument passed in
    returning an atoms object.Note that only one argument may be passed
    in at a time.

    Parameters:
        see `ase.data.pubchem.pubchem_search`

    returns:
        atoms (ASE Atoms Object):
            an ASE Atoms object containing the information on the
            requested entry
    """
    return pubchem_search(*args, **kwargs).get_atoms()


def pubchem_atoms_conformer_search(*args, **kwargs):
    """
    Search PubChem for all the conformers of a given compound.
    Note that only one argument may be passed in at a time.

    Parameters:
        see `ase.data.pubchem.pubchem_search`

    returns:
        conformers (list):
            a list containing the atoms objects of all the conformers
            for your search
    """
    conformers = pubchem_conformer_search(*args, **kwargs)
    conformers = [conformer.get_atoms() for conformer in conformers]
    return conformers
