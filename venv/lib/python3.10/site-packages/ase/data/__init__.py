# fmt: off

import numpy as np

from ase.data.vdw import vdw_radii

__all__ = ['vdw_radii', 'chemical_symbols', 'ground_state_magnetic_moments',
           'reference_states', 'atomic_names', 'atomic_masses',
           'atomic_numbers', 'covalent_radii']

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

atomic_numbers = {symbol: Z for Z, symbol in enumerate(chemical_symbols)}
# IUPAC version dated 28 November 2016
atomic_names = [
    '', 'Hydrogen', 'Helium', 'Lithium', 'Beryllium', 'Boron',
    'Carbon', 'Nitrogen', 'Oxygen', 'Fluorine', 'Neon', 'Sodium',
    'Magnesium', 'Aluminium', 'Silicon', 'Phosphorus', 'Sulfur',
    'Chlorine', 'Argon', 'Potassium', 'Calcium', 'Scandium',
    'Titanium', 'Vanadium', 'Chromium', 'Manganese', 'Iron',
    'Cobalt', 'Nickel', 'Copper', 'Zinc', 'Gallium', 'Germanium',
    'Arsenic', 'Selenium', 'Bromine', 'Krypton', 'Rubidium',
    'Strontium', 'Yttrium', 'Zirconium', 'Niobium', 'Molybdenum',
    'Technetium', 'Ruthenium', 'Rhodium', 'Palladium', 'Silver',
    'Cadmium', 'Indium', 'Tin', 'Antimony', 'Tellurium',
    'Iodine', 'Xenon', 'Caesium', 'Barium', 'Lanthanum',
    'Cerium', 'Praseodymium', 'Neodymium', 'Promethium',
    'Samarium', 'Europium', 'Gadolinium', 'Terbium',
    'Dysprosium', 'Holmium', 'Erbium', 'Thulium', 'Ytterbium',
    'Lutetium', 'Hafnium', 'Tantalum', 'Tungsten', 'Rhenium',
    'Osmium', 'Iridium', 'Platinum', 'Gold', 'Mercury',
    'Thallium', 'Lead', 'Bismuth', 'Polonium', 'Astatine',
    'Radon', 'Francium', 'Radium', 'Actinium', 'Thorium',
    'Protactinium', 'Uranium', 'Neptunium', 'Plutonium',
    'Americium', 'Curium', 'Berkelium', 'Californium',
    'Einsteinium', 'Fermium', 'Mendelevium', 'Nobelium',
    'Lawrencium', 'Rutherfordium', 'Dubnium', 'Seaborgium',
    'Bohrium', 'Hassium', 'Meitnerium', 'Darmastadtium',
    'Roentgenium', 'Copernicium', 'Nihonium', 'Flerovium',
    'Moscovium', 'Livermorium', 'Tennessine', 'Oganesson']

# Atomic masses are based on:
#
#   Meija, J., Coplen, T., Berglund, M., et al. (2016). Atomic weights of
#   the elements 2013 (IUPAC Technical Report). Pure and Applied Chemistry,
#   88(3), pp. 265-291. Retrieved 30 Nov. 2016,
#   from doi:10.1515/pac-2015-0305
#
# Standard atomic weights are taken from Table 1: "Standard atomic weights
# 2013", with the uncertainties ignored.
# For hydrogen, helium, boron, carbon, nitrogen, oxygen, magnesium, silicon,
# sulfur, chlorine, bromine and thallium, where the weights are given as a
# range the "conventional" weights are taken from Table 3 and the ranges are
# given in the comments.
# The mass of the most stable isotope (in Table 4) is used for elements
# where there the element has no stable isotopes (to avoid NaNs): Tc, Pm,
# Po, At, Rn, Fr, Ra, Ac, everything after Np
atomic_masses_iupac2016 = np.array([
    1.0,  # X
    1.008,  # H [1.00784, 1.00811]
    4.002602,  # He
    6.94,  # Li [6.938, 6.997]
    9.0121831,  # Be
    10.81,  # B [10.806, 10.821]
    12.011,  # C [12.0096, 12.0116]
    14.007,  # N [14.00643, 14.00728]
    15.999,  # O [15.99903, 15.99977]
    18.998403163,  # F
    20.1797,  # Ne
    22.98976928,  # Na
    24.305,  # Mg [24.304, 24.307]
    26.9815385,  # Al
    28.085,  # Si [28.084, 28.086]
    30.973761998,  # P
    32.06,  # S [32.059, 32.076]
    35.45,  # Cl [35.446, 35.457]
    39.948,  # Ar
    39.0983,  # K
    40.078,  # Ca
    44.955908,  # Sc
    47.867,  # Ti
    50.9415,  # V
    51.9961,  # Cr
    54.938044,  # Mn
    55.845,  # Fe
    58.933194,  # Co
    58.6934,  # Ni
    63.546,  # Cu
    65.38,  # Zn
    69.723,  # Ga
    72.630,  # Ge
    74.921595,  # As
    78.971,  # Se
    79.904,  # Br [79.901, 79.907]
    83.798,  # Kr
    85.4678,  # Rb
    87.62,  # Sr
    88.90584,  # Y
    91.224,  # Zr
    92.90637,  # Nb
    95.95,  # Mo
    97.90721,  # 98Tc
    101.07,  # Ru
    102.90550,  # Rh
    106.42,  # Pd
    107.8682,  # Ag
    112.414,  # Cd
    114.818,  # In
    118.710,  # Sn
    121.760,  # Sb
    127.60,  # Te
    126.90447,  # I
    131.293,  # Xe
    132.90545196,  # Cs
    137.327,  # Ba
    138.90547,  # La
    140.116,  # Ce
    140.90766,  # Pr
    144.242,  # Nd
    144.91276,  # 145Pm
    150.36,  # Sm
    151.964,  # Eu
    157.25,  # Gd
    158.92535,  # Tb
    162.500,  # Dy
    164.93033,  # Ho
    167.259,  # Er
    168.93422,  # Tm
    173.054,  # Yb
    174.9668,  # Lu
    178.49,  # Hf
    180.94788,  # Ta
    183.84,  # W
    186.207,  # Re
    190.23,  # Os
    192.217,  # Ir
    195.084,  # Pt
    196.966569,  # Au
    200.592,  # Hg
    204.38,  # Tl [204.382, 204.385]
    207.2,  # Pb
    208.98040,  # Bi
    208.98243,  # 209Po
    209.98715,  # 210At
    222.01758,  # 222Rn
    223.01974,  # 223Fr
    226.02541,  # 226Ra
    227.02775,  # 227Ac
    232.0377,  # Th
    231.03588,  # Pa
    238.02891,  # U
    237.04817,  # 237Np
    244.06421,  # 244Pu
    243.06138,  # 243Am
    247.07035,  # 247Cm
    247.07031,  # 247Bk
    251.07959,  # 251Cf
    252.0830,  # 252Es
    257.09511,  # 257Fm
    258.09843,  # 258Md
    259.1010,  # 259No
    262.110,  # 262Lr
    267.122,  # 267Rf
    268.126,  # 268Db
    271.134,  # 271Sg
    270.133,  # 270Bh
    269.1338,  # 269Hs
    278.156,  # 278Mt
    281.165,  # 281Ds
    281.166,  # 281Rg
    285.177,  # 285Cn
    286.182,  # 286Nh
    289.190,  # 289Fl
    289.194,  # 289Mc
    293.204,  # 293Lv
    293.208,  # 293Ts
    294.214,  # 294Og
])

# set atomic_masses to most recent version
atomic_masses = atomic_masses_iupac2016

atomic_masses_legacy = np.array([
    1.00000,  # X
    1.00794,  # H
    4.00260,  # He
    6.94100,  # Li
    9.01218,  # Be
    10.81100,  # B
    12.01100,  # C
    14.00670,  # N
    15.99940,  # O
    18.99840,  # F
    20.17970,  # Ne
    22.98977,  # Na
    24.30500,  # Mg
    26.98154,  # Al
    28.08550,  # Si
    30.97376,  # P
    32.06600,  # S
    35.45270,  # Cl
    39.94800,  # Ar
    39.09830,  # K
    40.07800,  # Ca
    44.95590,  # Sc
    47.88000,  # Ti
    50.94150,  # V
    51.99600,  # Cr
    54.93800,  # Mn
    55.84700,  # Fe
    58.93320,  # Co
    58.69340,  # Ni
    63.54600,  # Cu
    65.39000,  # Zn
    69.72300,  # Ga
    72.61000,  # Ge
    74.92160,  # As
    78.96000,  # Se
    79.90400,  # Br
    83.80000,  # Kr
    85.46780,  # Rb
    87.62000,  # Sr
    88.90590,  # Y
    91.22400,  # Zr
    92.90640,  # Nb
    95.94000,  # Mo
    np.nan,  # Tc
    101.07000,  # Ru
    102.90550,  # Rh
    106.42000,  # Pd
    107.86800,  # Ag
    112.41000,  # Cd
    114.82000,  # In
    118.71000,  # Sn
    121.75700,  # Sb
    127.60000,  # Te
    126.90450,  # I
    131.29000,  # Xe
    132.90540,  # Cs
    137.33000,  # Ba
    138.90550,  # La
    140.12000,  # Ce
    140.90770,  # Pr
    144.24000,  # Nd
    np.nan,  # Pm
    150.36000,  # Sm
    151.96500,  # Eu
    157.25000,  # Gd
    158.92530,  # Tb
    162.50000,  # Dy
    164.93030,  # Ho
    167.26000,  # Er
    168.93420,  # Tm
    173.04000,  # Yb
    174.96700,  # Lu
    178.49000,  # Hf
    180.94790,  # Ta
    183.85000,  # W
    186.20700,  # Re
    190.20000,  # Os
    192.22000,  # Ir
    195.08000,  # Pt
    196.96650,  # Au
    200.59000,  # Hg
    204.38300,  # Tl
    207.20000,  # Pb
    208.98040,  # Bi
    np.nan,  # Po
    np.nan,  # At
    np.nan,  # Rn
    np.nan,  # Fr
    226.02540,  # Ra
    np.nan,  # Ac
    232.03810,  # Th
    231.03590,  # Pa
    238.02900,  # U
    237.04820,  # Np
    np.nan,  # Pu
    np.nan,  # Am
    np.nan,  # Cm
    np.nan,  # Bk
    np.nan,  # Cf
    np.nan,  # Es
    np.nan,  # Fm
    np.nan,  # Md
    np.nan,  # No
    np.nan  # Lw
])

atomic_masses_common = np.array([
    1.0,             # X
    1.00782503223,   # H
    4.00260325413,   # He
    7.0160034366,    # Li
    9.012183065,     # Be
    11.00930536,     # B
    12.0000000,      # C
    14.00307400443,  # N
    15.99491461957,  # O
    18.99840316273,  # F
    19.9924401762,   # Ne
    22.9897692820,   # Na
    23.985041697,    # Mg
    26.98153853,     # Al
    27.97692653465,  # Si
    30.97376199842,  # P
    31.9720711744,   # S
    34.968852682,    # Cl
    39.9623831237,   # Ar
    38.9637064864,   # K
    39.962590863,    # Ca
    44.95590828,     # Sc
    47.94794198,     # Ti
    50.94395704,     # V
    51.94050623,     # Cr
    54.93804391,     # Mn
    55.93493633,     # Fe
    58.93319429,     # Co
    57.93534241,     # Ni
    62.92959772,     # Cu
    63.92914201,     # Zn
    68.9255735,      # Ga
    73.921177761,    # Ge
    74.92159457,     # As
    79.9165218,      # Se
    78.9183376,      # Br
    83.9114977282,   # Kr
    84.9117897379,   # Rb
    87.9056125,      # Sr
    88.9058403,      # Y
    89.9046977,      # Zr
    92.9063730,      # Nb
    97.90540482,     # Mo
    96.9063667,      # Tc
    101.9043441,     # Ru
    102.9054980,     # Rh
    105.9034804,     # Pd
    106.9050916,     # Ag
    113.90336509,    # Cd
    114.903878776,   # In
    119.90220163,    # Sn
    120.9038120,     # Sb
    129.906222748,   # Te
    126.9044719,     # I
    131.9041550856,  # Xe
    132.9054519610,  # Cs
    137.90524700,    # Ba
    138.9063563,     # La
    139.9054431,     # Ce
    140.9076576,     # Pr
    141.9077290,     # Nd
    144.9127559,     # Pm
    151.9197397,     # Sm
    152.9212380,     # Eu
    157.9241123,     # Gd
    158.9253547,     # Tb
    163.9291819,     # Dy
    164.9303288,     # Ho
    165.9302995,     # Er
    168.9342179,     # Tm
    173.9388664,     # Yb
    174.9407752,     # Lu
    179.9465570,     # Hf
    180.9479958,     # Ta
    183.95093092,    # W
    186.9557501,     # Re
    191.9614770,     # Os
    192.9629216,     # Ir
    194.9647917,     # Pt
    196.96656879,    # Au
    201.97064340,    # Hg
    204.9744278,     # Tl
    207.9766525,     # Pb
    208.9803991,     # Bi
    208.9824308,     # Po
    209.9871479,     # At
    222.0175782,     # Rn
    223.0197360,     # Fr
    226.0254103,     # Ra
    227.0277523,     # Ac
    232.0380558,     # Th
    231.0358842,     # Pa
    238.0507884,     # U
    237.0481736,     # Np
    244.0642053,     # Pu
    243.0613813,     # Am
    247.0703541,     # Cm
    247.0703073,     # Bk
    251.0795886,     # Cf
    252.082980,      # Es
    257.0951061,     # Fm
    258.0984315,     # Md
    259.10103,       # No
    262.10961,       # Lr
    267.12179,       # Rf
    268.12567,       # Db
    271.13393,       # Sg
    272.13826,       # Bh
    270.13429,       # Hs
    276.15159,       # Mt
    281.16451,       # Ds
    280.16514,       # Rg
    285.17712,       # Cn
    284.17873,       # Nh
    289.19042,       # Fl
    288.19274,       # Mc
    293.20449,       # Lv
    292.20746,       # Ts
    294.21392,       # Og
])


# Covalent radii from:
#
#  Covalent radii revisited,
#  Beatriz Cordero, Verónica Gómez, Ana E. Platero-Prats, Marc Revés,
#  Jorge Echeverría, Eduard Cremades, Flavia Barragán and Santiago Alvarez,
#  Dalton Trans., 2008, 2832-2838 DOI:10.1039/B801115J
missing = 0.2
covalent_radii = np.array([
    missing,  # X
    0.31,  # H
    0.28,  # He
    1.28,  # Li
    0.96,  # Be
    0.84,  # B
    0.76,  # C
    0.71,  # N
    0.66,  # O
    0.57,  # F
    0.58,  # Ne
    1.66,  # Na
    1.41,  # Mg
    1.21,  # Al
    1.11,  # Si
    1.07,  # P
    1.05,  # S
    1.02,  # Cl
    1.06,  # Ar
    2.03,  # K
    1.76,  # Ca
    1.70,  # Sc
    1.60,  # Ti
    1.53,  # V
    1.39,  # Cr
    1.39,  # Mn
    1.32,  # Fe
    1.26,  # Co
    1.24,  # Ni
    1.32,  # Cu
    1.22,  # Zn
    1.22,  # Ga
    1.20,  # Ge
    1.19,  # As
    1.20,  # Se
    1.20,  # Br
    1.16,  # Kr
    2.20,  # Rb
    1.95,  # Sr
    1.90,  # Y
    1.75,  # Zr
    1.64,  # Nb
    1.54,  # Mo
    1.47,  # Tc
    1.46,  # Ru
    1.42,  # Rh
    1.39,  # Pd
    1.45,  # Ag
    1.44,  # Cd
    1.42,  # In
    1.39,  # Sn
    1.39,  # Sb
    1.38,  # Te
    1.39,  # I
    1.40,  # Xe
    2.44,  # Cs
    2.15,  # Ba
    2.07,  # La
    2.04,  # Ce
    2.03,  # Pr
    2.01,  # Nd
    1.99,  # Pm
    1.98,  # Sm
    1.98,  # Eu
    1.96,  # Gd
    1.94,  # Tb
    1.92,  # Dy
    1.92,  # Ho
    1.89,  # Er
    1.90,  # Tm
    1.87,  # Yb
    1.87,  # Lu
    1.75,  # Hf
    1.70,  # Ta
    1.62,  # W
    1.51,  # Re
    1.44,  # Os
    1.41,  # Ir
    1.36,  # Pt
    1.36,  # Au
    1.32,  # Hg
    1.45,  # Tl
    1.46,  # Pb
    1.48,  # Bi
    1.40,  # Po
    1.50,  # At
    1.50,  # Rn
    2.60,  # Fr
    2.21,  # Ra
    2.15,  # Ac
    2.06,  # Th
    2.00,  # Pa
    1.96,  # U
    1.90,  # Np
    1.87,  # Pu
    1.80,  # Am
    1.69,  # Cm
    missing,  # Bk
    missing,  # Cf
    missing,  # Es
    missing,  # Fm
    missing,  # Md
    missing,  # No
    missing,  # Lr
    missing,  # Rf
    missing,  # Db
    missing,  # Sg
    missing,  # Bh
    missing,  # Hs
    missing,  # Mt
    missing,  # Ds
    missing,  # Rg
    missing,  # Cn
    missing,  # Nh
    missing,  # Fl
    missing,  # Mc
    missing,  # Lv
    missing,  # Ts
    missing,  # Og
])


# This data is from Ashcroft and Mermin.
# Most constants are listed in periodic table, inside front cover.
# Reference states that have a non-trivial basis have a 'basis' key.
# If the basis is None, it means it has a basis but we have not tabulated it.
# For basis of RHL systems (represented here as basis_x) see page 127.
# For TET systems see page 127, too.
#
# (Values for magnetic moments are not from A&M and indeed are not taken
# from anywhere in particular.)
reference_states = [
    None,  # X
    {'symmetry': 'diatom', 'd': 0.74},  # H
    {'symmetry': 'atom'},  # He
    {'symmetry': 'bcc', 'a': 3.49},  # Li
    {'symmetry': 'hcp', 'c/a': 1.567, 'a': 2.29},  # Be
    {'symmetry': 'tetragonal', 'c/a': 0.576, 'a': 8.73,  # B
     'basis': None},
    {'symmetry': 'diamond', 'a': 3.57},  # C
    {'symmetry': 'diatom', 'd': 1.10},  # N
    {'symmetry': 'diatom', 'd': 1.21},  # O
    {'symmetry': 'diatom', 'd': 1.42},  # F
    {'symmetry': 'fcc', 'a': 4.43},  # Ne
    {'symmetry': 'bcc', 'a': 4.23},  # Na
    {'symmetry': 'hcp', 'c/a': 1.624, 'a': 3.21},  # Mg
    {'symmetry': 'fcc', 'a': 4.05},  # Al
    {'symmetry': 'diamond', 'a': 5.43},  # Si
    {'symmetry': 'cubic', 'a': 7.17,  # P
     'basis': None},
    {'symmetry': 'orthorhombic', 'c/a': 2.339, 'a': 10.47, 'b/a': 1.229,  # S
     'basis': None},
    {'symmetry': 'orthorhombic', 'c/a': 1.324, 'a': 6.24, 'b/a': 0.718,  # Cl
     'basis': None},
    {'symmetry': 'fcc', 'a': 5.26},  # Ar
    {'symmetry': 'bcc', 'a': 5.23},  # K
    {'symmetry': 'fcc', 'a': 5.58},  # Ca
    {'symmetry': 'hcp', 'c/a': 1.594, 'a': 3.31},  # Sc
    {'symmetry': 'hcp', 'c/a': 1.588, 'a': 2.95},  # Ti
    {'symmetry': 'bcc', 'a': 3.02},  # V
    {'symmetry': 'bcc', 'a': 2.88},  # Cr
    {'symmetry': 'cubic', 'a': 8.89,  # Mn
     'basis': None},
    {'symmetry': 'bcc', 'a': 2.87, 'magmom_per_atom': 2.3},  # Fe
    {'symmetry': 'hcp', 'c/a': 1.622, 'a': 2.51, 'magmom_per_atom': 1.2},  # Co
    {'symmetry': 'fcc', 'a': 3.52, 'magmom_per_atom': 0.6},  # Ni
    {'symmetry': 'fcc', 'a': 3.61},  # Cu
    {'symmetry': 'hcp', 'c/a': 1.856, 'a': 2.66},  # Zn
    {'symmetry': 'orthorhombic', 'c/a': 1.695, 'a': 4.51, 'b/a': 1.001,  # Ga
     'basis': None},
    {'symmetry': 'diamond', 'a': 5.66},  # Ge
    {'symmetry': 'rhombohedral', 'a': 4.13, 'alpha': 54.10,  # As
     'basis_x': np.array(0.226) * (-1, 1)},
    {'symmetry': 'hcp', 'c/a': 1.136, 'a': 4.36,  # Se
     'basis': None},  # Needs 3-atom basis
    {'symmetry': 'orthorhombic', 'c/a': 1.307, 'a': 6.67, 'b/a': 0.672,  # Br
     'basis': None},
    {'symmetry': 'fcc', 'a': 5.72},  # Kr
    {'symmetry': 'bcc', 'a': 5.59},  # Rb
    {'symmetry': 'fcc', 'a': 6.08},  # Sr
    {'symmetry': 'hcp', 'c/a': 1.571, 'a': 3.65},  # Y
    {'symmetry': 'hcp', 'c/a': 1.593, 'a': 3.23},  # Zr
    {'symmetry': 'bcc', 'a': 3.30},  # Nb
    {'symmetry': 'bcc', 'a': 3.15},  # Mo
    {'symmetry': 'hcp', 'c/a': 1.604, 'a': 2.74},  # Tc
    {'symmetry': 'hcp', 'c/a': 1.584, 'a': 2.70},  # Ru
    {'symmetry': 'fcc', 'a': 3.80},  # Rh
    {'symmetry': 'fcc', 'a': 3.89},  # Pd
    {'symmetry': 'fcc', 'a': 4.09},  # Ag
    {'symmetry': 'hcp', 'c/a': 1.886, 'a': 2.98},  # Cd
    # For In, A&M give a face-centered cell; we need some sqrt2 conversions.
    {'symmetry': 'bct', 'c/a': 1.076 * 2**.5, 'a': 4.59 / 2**.5},  # In
    {'symmetry': 'bct', 'c/a': 0.546, 'a': 5.82,  # Sn
     'basis': [[0.0, 0.0, 0.0], [0.25, 0.75, 0.5]]},
    {'symmetry': 'rhombohedral', 'a': 4.51, 'alpha': 57.60,  # Sb
     'basis_x': np.array(0.233) * (-1, 1)},
    {'symmetry': 'hcp', 'c/a': 1.330, 'a': 4.45,  # Te
     'basis': None},  # Te needs a 3-atom basis.
    {'symmetry': 'orthorhombic', 'c/a': 1.347, 'a': 7.27, 'b/a': 0.659,  # I
     'basis': None},
    {'symmetry': 'fcc', 'a': 6.20},  # Xe
    {'symmetry': 'bcc', 'a': 6.05},  # Cs
    {'symmetry': 'bcc', 'a': 5.02},  # Ba
    {'symmetry': 'hcp', 'c/a': 1.619, 'a': 3.75},  # La
    {'symmetry': 'fcc', 'a': 5.16},  # Ce
    {'symmetry': 'hcp', 'c/a': 1.614, 'a': 3.67},  # Pr
    {'symmetry': 'hcp', 'c/a': 1.614, 'a': 3.66},  # Nd
    None,  # Pm
    {'symmetry': 'rhombohedral', 'a': 9.00, 'alpha': 23.13,
     'basis_x': np.array(0.222) * (0, -1, 1)},  # Sm
    {'symmetry': 'bcc', 'a': 4.61},  # Eu
    {'symmetry': 'hcp', 'c/a': 1.588, 'a': 3.64},  # Gd
    {'symmetry': 'hcp', 'c/a': 1.581, 'a': 3.60},  # Th
    {'symmetry': 'hcp', 'c/a': 1.573, 'a': 3.59},  # Dy
    {'symmetry': 'hcp', 'c/a': 1.570, 'a': 3.58},  # Ho
    {'symmetry': 'hcp', 'c/a': 1.570, 'a': 3.56},  # Er
    {'symmetry': 'hcp', 'c/a': 1.570, 'a': 3.54},  # Tm
    {'symmetry': 'fcc', 'a': 5.49},  # Yb
    {'symmetry': 'hcp', 'c/a': 1.585, 'a': 3.51},  # Lu
    {'symmetry': 'hcp', 'c/a': 1.582, 'a': 3.20},  # Hf
    {'symmetry': 'bcc', 'a': 3.31},  # Ta
    {'symmetry': 'bcc', 'a': 3.16},  # W
    {'symmetry': 'hcp', 'c/a': 1.615, 'a': 2.76},  # Re
    {'symmetry': 'hcp', 'c/a': 1.579, 'a': 2.74},  # Os
    {'symmetry': 'fcc', 'a': 3.84},  # Ir
    {'symmetry': 'fcc', 'a': 3.92},  # Pt
    {'symmetry': 'fcc', 'a': 4.08},  # Au
    {'symmetry': 'rhombohedral', 'a': 2.99, 'alpha': 70.45,  # Hg
     'basis_x': np.zeros(1)},
    {'symmetry': 'hcp', 'c/a': 1.599, 'a': 3.46},  # Tl
    {'symmetry': 'fcc', 'a': 4.95},  # Pb
    {'symmetry': 'rhombohedral', 'a': 4.75, 'alpha': 57.14,
     'basis_x': np.array(0.237) * (-1, 1)},  # Bi
    {'symmetry': 'sc', 'a': 3.35},  # Po
    None,  # At
    None,  # Rn
    None,  # Fr
    None,  # Ra
    {'symmetry': 'fcc', 'a': 5.31},  # Ac
    {'symmetry': 'fcc', 'a': 5.08},  # Th
    {'symmetry': 'tetragonal', 'c/a': 0.825, 'a': 3.92},  # Pa
    {'symmetry': 'orthorhombic', 'c/a': 2.056, 'a': 2.85, 'b/a': 1.736},  # U
    {'symmetry': 'orthorhombic', 'c/a': 1.411, 'a': 4.72, 'b/a': 1.035},  # Np
    {'symmetry': 'monoclinic'},  # Pu
    None,  # Am
    None,  # Cm
    None,  # Bk
    None,  # Cf
    None,  # Es
    None,  # Fm
    None,  # Md
    None,  # No
    None,  # Lr
    None,  # Rf
    None,  # Db
    None,  # Sg
    None,  # Bh
    None,  # Hs
    None,  # Mt
    None,  # Ds
    None,  # Rg
    None,  # Cn
    None,  # Nh
    None,  # Fl
    None,  # Mc
    None,  # Lv
    None,  # Ts
    None,  # Og
]

# http://www.webelements.com
ground_state_magnetic_moments = np.array([
    0.0,  # X
    1.0,  # H
    0.0,  # He
    1.0,  # Li
    0.0,  # Be
    1.0,  # B
    2.0,  # C
    3.0,  # N
    2.0,  # O
    1.0,  # F
    0.0,  # Ne
    1.0,  # Na
    0.0,  # Mg
    1.0,  # Al
    2.0,  # Si
    3.0,  # P
    2.0,  # S
    1.0,  # Cl
    0.0,  # Ar
    1.0,  # K
    0.0,  # Ca
    1.0,  # Sc
    2.0,  # Ti
    3.0,  # V
    6.0,  # Cr
    5.0,  # Mn
    4.0,  # Fe
    3.0,  # Co
    2.0,  # Ni
    1.0,  # Cu
    0.0,  # Zn
    1.0,  # Ga
    2.0,  # Ge
    3.0,  # As
    2.0,  # Se
    1.0,  # Br
    0.0,  # Kr
    1.0,  # Rb
    0.0,  # Sr
    1.0,  # Y
    2.0,  # Zr
    5.0,  # Nb
    6.0,  # Mo
    5.0,  # Tc
    4.0,  # Ru
    3.0,  # Rh
    0.0,  # Pd
    1.0,  # Ag
    0.0,  # Cd
    1.0,  # In
    2.0,  # Sn
    3.0,  # Sb
    2.0,  # Te
    1.0,  # I
    0.0,  # Xe
    1.0,  # Cs
    0.0,  # Ba
    1.0,  # La
    1.0,  # Ce
    3.0,  # Pr
    4.0,  # Nd
    5.0,  # Pm
    6.0,  # Sm
    7.0,  # Eu
    8.0,  # Gd
    5.0,  # Tb
    4.0,  # Dy
    3.0,  # Ho
    2.0,  # Er
    1.0,  # Tm
    0.0,  # Yb
    1.0,  # Lu
    2.0,  # Hf
    3.0,  # Ta
    4.0,  # W
    5.0,  # Re
    4.0,  # Os
    3.0,  # Ir
    2.0,  # Pt
    1.0,  # Au
    0.0,  # Hg
    1.0,  # Tl
    2.0,  # Pb
    3.0,  # Bi
    2.0,  # Po
    1.0,  # At
    0.0,  # Rn
    1.0,  # Fr
    0.0,  # Ra
    1.0,  # Ac
    2.0,  # Th
    3.0,  # Pa
    4.0,  # U
    5.0,  # Np
    6.0,  # Pu
    7.0,  # Am
    8.0,  # Cm
    5.0,  # Bk
    4.0,  # Cf
    4.0,  # Es
    2.0,  # Fm
    1.0,  # Md
    0.0,  # No
    np.nan])  # Lr
