# fmt: off

import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

from ase import Atoms
from ase.utils import writer


def read_xsd(fd):
    tree = ET.parse(fd)
    root = tree.getroot()

    atomtreeroot = root.find('AtomisticTreeRoot')
    # if periodic system
    if atomtreeroot.find('SymmetrySystem') is not None:
        symmetrysystem = atomtreeroot.find('SymmetrySystem')
        mappingset = symmetrysystem.find('MappingSet')
        mappingfamily = mappingset.find('MappingFamily')
        system = mappingfamily.find('IdentityMapping')

        coords = []
        cell = []
        formula = ''

        for atom in system:
            if atom.tag == 'Atom3d':
                symbol = atom.get('Components')
                formula += symbol

                xyz = atom.get('XYZ')
                if xyz:
                    coord = [float(coord) for coord in xyz.split(',')]
                else:
                    coord = [0.0, 0.0, 0.0]
                coords.append(coord)
            elif atom.tag == 'SpaceGroup':
                avec = [float(vec) for vec in atom.get('AVector').split(',')]
                bvec = [float(vec) for vec in atom.get('BVector').split(',')]
                cvec = [float(vec) for vec in atom.get('CVector').split(',')]

                cell.append(avec)
                cell.append(bvec)
                cell.append(cvec)

        atoms = Atoms(formula, cell=cell, pbc=True)
        atoms.set_scaled_positions(coords)
        return atoms
        # if non-periodic system
    elif atomtreeroot.find('Molecule') is not None:
        system = atomtreeroot.find('Molecule')

        coords = []
        formula = ''

        for atom in system:
            if atom.tag == 'Atom3d':
                symbol = atom.get('Components')
                formula += symbol

                xyz = atom.get('XYZ')
                coord = [float(coord) for coord in xyz.split(',')]
                coords.append(coord)

        atoms = Atoms(formula, pbc=False)
        atoms.set_scaled_positions(coords)
        return atoms


def CPK_or_BnS(element):
    """Determine how atom is visualized"""
    if element in ['C', 'H', 'O', 'S', 'N']:
        visualization_choice = 'Ball and Stick'
    else:
        visualization_choice = 'CPK'
    return visualization_choice


def SetChild(parent, childname, props):
    Child = ET.SubElement(parent, childname)
    for key in props:
        Child.set(key, props[key])
    return Child


def SetBasicChilds():
    """
    Basic property setup for Material Studio File
    """
    XSD = ET.Element('XSD')
    XSD.set('Version', '6.0')

    ATR = SetChild(XSD, 'AtomisticTreeRoot', dict(
        ID='1',
        NumProperties='40',
        NumChildren='1',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='AngleEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='BendBendEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='BendTorsionBendEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='BondEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='Atom',
        Name='EFGAsymmetry',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='Atom',
        Name='EFGQuadrupolarCoupling',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='ElectrostaticEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='GrowthFace',
        Name='FaceMillerIndex',
        Type='MillerIndex',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='GrowthFace',
        Name='FacetTransparency',
        Type='Float',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='Bondable',
        Name='Force',
        Type='CoDirection',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='HydrogenBondEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='Bondable',
        Name='ImportOrder',
        Type='UnsignedInteger',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='InversionEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='Atom',
        Name='IsBackboneAtom',
        Type='Boolean',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='Atom',
        Name='IsChiralCenter',
        Type='Boolean',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='Atom',
        Name='IsOutOfPlane',
        Type='Boolean',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='BestFitLineMonitor',
        Name='LineExtentPadding',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='Linkage',
        Name='LinkageGroupName',
        Type='String',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='PropertyList',
        Name='ListIdentifier',
        Type='String',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='Atom',
        Name='NMRShielding',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='NonBondEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='Bondable',
        Name='NormalMode',
        Type='Direction',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='Bondable',
        Name='NormalModeFrequency',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='Bondable',
        Name='OrbitalCutoffRadius',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='BestFitPlaneMonitor',
        Name='PlaneExtentPadding',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='PotentialEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ScalarFieldBase',
        Name='QuantizationValue',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='RestraintEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='SeparatedStretchStretchEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='Trajectory',
        Name='SimulationStep',
        Type='Integer',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='StretchBendStretchEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='StretchStretchEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='StretchTorsionStretchEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='TorsionBendBendEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='TorsionEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='TorsionStretchEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='ValenceCrossTermEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='ValenceDiagonalEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='ClassicalEnergyHolder',
        Name='VanDerWaalsEnergy',
        Type='Double',
    ))
    SetChild(ATR, 'Property', dict(
        DefinedOn='SymmetrySystem',
        Name='_Stress',
        Type='Matrix',
    ))
    return ATR, XSD


def _write_xsd_html(images, connectivity=None):
    ATR, XSD = SetBasicChilds()
    natoms = len(images[0])
    atom_element = images[0].get_chemical_symbols()
    atom_cell = images[0].get_cell()
    atom_positions = images[0].get_positions()
    # Set up bonds
    bonds = []
    if connectivity is not None:
        for i in range(connectivity.shape[0]):
            for j in range(i + 1, connectivity.shape[0]):
                if connectivity[i, j]:
                    bonds.append([i, j])
    nbonds = len(bonds)

    # non-periodic system
    if not images[0].pbc.all():
        Molecule = SetChild(ATR, 'Molecule', dict(
            ID='2',
            NumChildren=str(natoms + nbonds),
            Name='Lattice=&quot1.0',
        ))
        # writing images[0]
        for x in range(natoms):
            Props = dict(
                ID=str(x + 3),
                Name=(atom_element[x] + str(x + 1)),
                UserID=str(x + 1),
                DisplayStyle=CPK_or_BnS(atom_element[x]),
                XYZ=','.join('%1.16f' % xi for xi in atom_positions[x]),
                Components=atom_element[x],
            )
            bondstr = []
            for i, bond in enumerate(bonds):
                if x in bond:
                    bondstr.append(str(i + 3 + natoms))
            if bondstr:
                Props['Connections'] = ','.join(bondstr)
            SetChild(Molecule, 'Atom3d', Props)
        for x in range(nbonds):
            SetChild(Molecule, 'Bond', dict(
                ID=str(x + 3 + natoms),
                Connects='%i,%i' % (bonds[x][0] + 3, bonds[x][1] + 3),
            ))
    # periodic system
    else:
        atom_positions = np.dot(atom_positions, np.linalg.inv(atom_cell))
        Props = dict(
            ID='2',
            Mapping='3',
            Children=','.join(map(str, range(4, natoms + nbonds + 5))),
            Normalized='1',
            Name='SymmSys',
            UserID=str(natoms + 18),
            XYZ='0.00000000000000,0.00000000000000,0.000000000000000',
            OverspecificationTolerance='0.05',
            PeriodicDisplayType='Original',
        )
        SymmSys = SetChild(ATR, 'SymmetrySystem', Props)

        Props = dict(
            ID=str(natoms + nbonds + 5),
            SymmetryDefinition=str(natoms + 4),
            ActiveSystem='2',
            NumFamilies='1',
            OwnsTotalConstraintMapping='1',
            TotalConstraintMapping='3',
        )
        MappngSet = SetChild(SymmSys, 'MappingSet', Props)

        Props = dict(ID=str(natoms + nbonds + 6), NumImageMappings='0')
        MappngFamily = SetChild(MappngSet, 'MappingFamily', Props)

        Props = dict(
            ID=str(natoms + len(bonds) + 7),
            Element='1,0,0,0,0,1,0,0,0,0,1,0',
            Constraint='1,0,0,0,0,1,0,0,0,0,1,0',
            MappedObjects=','.join(map(str, range(4, natoms + len(bonds) + 4))),
            DefectObjects='%i,%i' % (natoms + nbonds + 4, natoms + nbonds + 8),
            NumImages=str(natoms + len(bonds)),
            NumDefects='2',
        )
        IdentMappng = SetChild(MappngFamily, 'IdentityMapping', Props)

        SetChild(MappngFamily, 'MappingRepairs', {'NumRepairs': '0'})

        # writing atoms
        for x in range(natoms):
            Props = dict(
                ID=str(x + 4),
                Mapping=str(natoms + len(bonds) + 7),
                Parent='2',
                Name=(atom_element[x] + str(x + 1)),
                UserID=str(x + 1),
                DisplayStyle=CPK_or_BnS(atom_element[x]),
                Components=atom_element[x],
                XYZ=','.join(['%1.16f' % xi for xi in atom_positions[x]]),
            )
            bondstr = []
            for i, bond in enumerate(bonds):
                if x in bond:
                    bondstr.append(str(i + 4 * natoms + 1))
            if bondstr:
                Props['Connections'] = ','.join(bondstr)
            SetChild(IdentMappng, 'Atom3d', Props)

        for x in range(len(bonds)):
            SetChild(IdentMappng, 'Bond', dict(
                ID=str(x + 4 + natoms + 1),
                Mapping=str(natoms + len(bonds) + 7),
                Parent='2',
                Connects='%i,%i' % (bonds[x][0] + 4, bonds[x][1] + 4),
            ))

        Props = dict(
            ID=str(natoms + 4),
            Parent='2',
            Children=str(natoms + len(bonds) + 8),
            DisplayStyle='Solid',
            XYZ='0.00,0.00,0.00',
            Color='0,0,0,0',
            AVector=','.join(['%1.16f' % atom_cell[0, x] for x in range(3)]),
            BVector=','.join(['%1.16f' % atom_cell[1, x] for x in range(3)]),
            CVector=','.join(['%1.16f' % atom_cell[2, x] for x in range(3)]),
            OrientationBase='C along Z, B in YZ plane',
            Centering='3D Primitive-Centered',
            Lattice='3D Triclinic',
            GroupName='GroupName',
            Operators='1,0,0,0,0,1,0,0,0,0,1,0',
            DisplayRange='0,1,0,1,0,1',
            LineThickness='2',
            CylinderRadius='0.2',
            LabelAxes='1',
            ActiveSystem='2',
            ITNumber='1',
            LongName='P 1',
            Qualifier='Origin-1',
            SchoenfliesName='C1-1',
            System='Triclinic',
            Class='1',
        )
        SetChild(IdentMappng, 'SpaceGroup', Props)

        SetChild(IdentMappng, 'ReciprocalLattice3D', dict(
            ID=str(natoms + len(bonds) + 8),
            Parent=str(natoms + 4),
        ))

        SetChild(MappngSet, 'InfiniteMapping', dict(
            ID='3',
            Element='1,0,0,0,0,1,0,0,0,0,1,0',
            MappedObjects='2',
        ))

    return XSD, ATR


@writer
def write_xsd(fd, images, connectivity=None):
    """Takes Atoms object, and write materials studio file
    atoms: Atoms object
    filename: path of the output file
    connectivity: number of atoms by number of atoms matrix for connectivity
    between atoms (0 not connected, 1 connected)

    note: material studio file cannot use a partial periodic system. If partial
    perodic system was inputted, full periodicity was assumed.
    """
    if hasattr(images, 'get_positions'):
        images = [images]

    XSD, _ATR = _write_xsd_html(images, connectivity)

    # Return a pretty-printed XML string for the Element.
    rough_string = ET.tostring(XSD, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    Document = reparsed.toprettyxml(indent='\t')

    fd.write(Document)
