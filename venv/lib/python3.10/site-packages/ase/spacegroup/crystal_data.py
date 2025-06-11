# fmt: off

from ase.lattice import bravais_classes

_crystal_family = ('Øaammmmmmmmmmmmmoooooooooooooooooooooooooooooooooooooooooo'
                   'ooooooooooooooooottttttttttttttttttttttttttttttttttttttttt'
                   'ttttttttttttttttttttttttttthhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh'
                   'hhhhhhhhhhhhhhhhhhhhhcccccccccccccccccccccccccccccccccccc')

_lattice_centering = ('ØPPPPCPPCCPPCPPCPPPPCCFIIPPPPPPPPPPCCCCCCCFFIIIPPPPPPPP'
                      'PPPPPPPPCCCCCCFFIIIIPPPPIIPIPPPPIIPPPPPPPPIIPPPPPPPPII'
                      'IIPPPPPPPPIIIIPPPPPPPPPPPPPPPPIIIIPPPRPRPPPPPPRPPPPRRP'
                      'PPPRRPPPPPPPPPPPPPPPPPPPPPPPPPPPPFIPIPPFFIPIPPFFIPPIPF'
                      'IPFIPPPPFFFFII')

_point_group_ranges = [(1, '1'),
                       (2, '-1'),
                       (3, '2'),
                       (6, 'm'),
                       (10, '2/m'),
                       (16, '222'),
                       (25, 'mm2'),
                       (47, '2/m 2/m 2/m'),
                       (75, '4'),
                       (81, '-4'),
                       (83, '4/m'),
                       (89, '422'),
                       (99, '4mm'),
                       (111, '-42m'),
                       (123, '4/m 2/m 2/m'),
                       (143, '3'),
                       (147, '-3'),
                       (149, '32'),
                       (156, '3m'),
                       (162, '-3 2/m'),
                       (168, '6'),
                       (174, '-6'),
                       (175, '6/m'),
                       (177, '622'),
                       (183, '6mm'),
                       (187, '-6m2'),
                       (191, '6/m 2/m 2/m'),
                       (195, '23'),
                       (200, '2/m -3'),
                       (207, '432'),
                       (215, '-43m'),
                       (221, '4/m -3 2/m'),
                       (231, 'Ø')]

_point_groups = ['Ø']
for i, (start, pg) in enumerate(_point_group_ranges[:-1]):
    next_start, _ = _point_group_ranges[i + 1]
    count = next_start - start
    for j in range(start, start + count):
        _point_groups.append(pg)


def validate_space_group(sg):
    sg = int(sg)
    if sg < 1:
        raise ValueError(f'Spacegroup must be positive, but is {sg}')
    if sg > 230:
        raise ValueError('Bad spacegroup', sg)
    return sg


def get_bravais_class(sg):
    sg = validate_space_group(sg)
    pearson_symbol = _crystal_family[sg] + _lattice_centering[sg]
    return bravais_classes[pearson_symbol]


def get_point_group(sg):
    sg = validate_space_group(sg)
    return _point_groups[sg]


def polar_space_group(sg):
    sg = validate_space_group(sg)
    pg = get_point_group(sg)
    return pg in ['1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm']
