# fmt: off

from ase.data import atomic_numbers, chemical_symbols, reference_states


def get_element_info(symbol, latticeconstant):
    if isinstance(symbol, str):
        atomic_number = atomic_numbers[symbol]
    else:
        atomic_number = symbol
        symbol = chemical_symbols[atomic_number]

    if latticeconstant is None:
        if reference_states[atomic_number]['symmetry'] in ['fcc', 'bcc', 'sc']:
            lattice_constant = reference_states[atomic_number]['a']
        else:
            raise NotImplementedError(
                "Cannot guess lattice constant of a %s element." %
                (reference_states[atomic_number]['symmetry'],))
    else:
        if isinstance(latticeconstant, (int, float)):
            lattice_constant = latticeconstant
        else:
            raise ValueError("Lattice constant must be of type int or float.")

    return symbol, atomic_number, lattice_constant
