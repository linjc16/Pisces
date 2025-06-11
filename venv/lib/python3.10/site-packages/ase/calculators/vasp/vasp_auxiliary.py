# fmt: off

import os
import re

import numpy as np


def get_vasp_version(string):
    """Extract version number from header of stdout.

    Example::

      >>> get_vasp_version('potato vasp.6.1.2 bumblebee')
      '6.1.2'

    """
    match = re.search(r'vasp\.(\S+)', string, re.M)
    return match.group(1)


class VaspChargeDensity:
    """Class for representing VASP charge density.

    Filename is normally CHG."""
    # Can the filename be CHGCAR?  There's a povray tutorial
    # in doc/tutorials where it's CHGCAR as of January 2021.  --askhl

    def __init__(self, filename):
        # Instance variables
        self.atoms = []  # List of Atoms objects
        self.chg = []  # Charge density
        self.chgdiff = []  # Charge density difference, if spin polarized
        self.aug = ''  # Augmentation charges, not parsed just a big string
        self.augdiff = ''  # Augmentation charge differece, is spin polarized

        # Note that the augmentation charge is not a list, since they
        # are needed only for CHGCAR files which store only a single
        # image.
        if filename is not None:
            self.read(filename)

    def is_spin_polarized(self):
        if len(self.chgdiff) > 0:
            return True
        return False

    def _read_chg(self, fobj, chg, volume):
        """Read charge from file object

        Utility method for reading the actual charge density (or
        charge density difference) from a file object. On input, the
        file object must be at the beginning of the charge block, on
        output the file position will be left at the end of the
        block. The chg array must be of the correct dimensions.

        """
        # VASP writes charge density as
        # WRITE(IU,FORM) (((C(NX,NY,NZ),NX=1,NGXC),NY=1,NGYZ),NZ=1,NGZC)
        # Fortran nested implied do loops; innermost index fastest
        # First, just read it in
        for zz in range(chg.shape[2]):
            for yy in range(chg.shape[1]):
                chg[:, yy, zz] = np.fromfile(fobj, count=chg.shape[0], sep=' ')
        chg /= volume

    def read(self, filename):
        """Read CHG or CHGCAR file.

        If CHG contains charge density from multiple steps all the
        steps are read and stored in the object. By default VASP
        writes out the charge density every 10 steps.

        chgdiff is the difference between the spin up charge density
        and the spin down charge density and is thus only read for a
        spin-polarized calculation.

        aug is the PAW augmentation charges found in CHGCAR. These are
        not parsed, they are just stored as a string so that they can
        be written again to a CHGCAR format file.

        """
        import ase.io.vasp as aiv
        with open(filename) as fd:
            self.atoms = []
            self.chg = []
            self.chgdiff = []
            self.aug = ''
            self.augdiff = ''
            while True:
                try:
                    atoms = aiv.read_vasp_configuration(fd)
                except (KeyError, RuntimeError, ValueError):
                    # Probably an empty line, or we tried to read the
                    # augmentation occupancies in CHGCAR
                    break

                # Note: We continue reading from the same file, and
                # this relies on read_vasp() to read no more lines
                # than it currently does.
                fd.readline()

                ngr = fd.readline().split()
                ng = (int(ngr[0]), int(ngr[1]), int(ngr[2]))
                chg = np.empty(ng)
                self._read_chg(fd, chg, atoms.get_volume())
                self.chg.append(chg)
                self.atoms.append(atoms)
                # Check if the file has a spin-polarized charge density part,
                # and if so, read it in.
                fl = fd.tell()
                # First check if the file has an augmentation charge part
                # (CHGCAR file.)
                line1 = fd.readline()
                if line1 == '':
                    break
                elif line1.find('augmentation') != -1:
                    augs = [line1]
                    while True:
                        line2 = fd.readline()
                        if line2.split() == ngr:
                            self.aug = ''.join(augs)
                            augs = []
                            chgdiff = np.empty(ng)
                            self._read_chg(fd, chgdiff, atoms.get_volume())
                            self.chgdiff.append(chgdiff)
                        elif line2 == '':
                            break
                        else:
                            augs.append(line2)
                    if len(self.aug) == 0:
                        self.aug = ''.join(augs)
                        augs = []
                    else:
                        self.augdiff = ''.join(augs)
                        augs = []
                elif line1.split() == ngr:
                    chgdiff = np.empty(ng)
                    self._read_chg(fd, chgdiff, atoms.get_volume())
                    self.chgdiff.append(chgdiff)
                else:
                    fd.seek(fl)

    def _write_chg(self, fobj, chg, volume, format='chg'):
        """Write charge density

        Utility function similar to _read_chg but for writing.

        """
        # Make a 1D copy of chg, must take transpose to get ordering right
        chgtmp = chg.T.ravel()
        # Multiply by volume
        chgtmp = chgtmp * volume
        # Must be a tuple to pass to string conversion
        chgtmp = tuple(chgtmp)
        # CHG format - 10 columns
        if format.lower() == 'chg':
            # Write all but the last row
            for ii in range((len(chgtmp) - 1) // 10):
                fobj.write(' %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G\
 %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G\n' % chgtmp[ii * 10:(ii + 1) * 10])
            # If the last row contains 10 values then write them without a
            # newline
            if len(chgtmp) % 10 == 0:
                fobj.write(' %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G'
                           ' %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G' %
                           chgtmp[len(chgtmp) - 10:len(chgtmp)])
            # Otherwise write fewer columns without a newline
            else:
                for ii in range(len(chgtmp) % 10):
                    fobj.write((' %#11.5G') %
                               chgtmp[len(chgtmp) - len(chgtmp) % 10 + ii])
        # Other formats - 5 columns
        else:
            # Write all but the last row
            for ii in range((len(chgtmp) - 1) // 5):
                fobj.write(' %17.10E %17.10E %17.10E %17.10E %17.10E\n' %
                           chgtmp[ii * 5:(ii + 1) * 5])
            # If the last row contains 5 values then write them without a
            # newline
            if len(chgtmp) % 5 == 0:
                fobj.write(' %17.10E %17.10E %17.10E %17.10E %17.10E' %
                           chgtmp[len(chgtmp) - 5:len(chgtmp)])
            # Otherwise write fewer columns without a newline
            else:
                for ii in range(len(chgtmp) % 5):
                    fobj.write((' %17.10E') %
                               chgtmp[len(chgtmp) - len(chgtmp) % 5 + ii])
        # Write a newline whatever format it is
        fobj.write('\n')

    def write(self, filename, format=None):
        """Write VASP charge density in CHG format.

        filename: str
            Name of file to write to.
        format: str
            String specifying whether to write in CHGCAR or CHG
            format.

        """
        import ase.io.vasp as aiv
        if format is None:
            if filename.lower().find('chgcar') != -1:
                format = 'chgcar'
            elif filename.lower().find('chg') != -1:
                format = 'chg'
            elif len(self.chg) == 1:
                format = 'chgcar'
            else:
                format = 'chg'
        with open(filename, 'w') as fd:
            for ii, chg in enumerate(self.chg):
                if format == 'chgcar' and ii != len(self.chg) - 1:
                    continue  # Write only the last image for CHGCAR
                aiv.write_vasp(fd,
                               self.atoms[ii],
                               direct=True)
                fd.write('\n')
                for dim in chg.shape:
                    fd.write(' %4i' % dim)
                fd.write('\n')
                vol = self.atoms[ii].get_volume()
                self._write_chg(fd, chg, vol, format)
                if format == 'chgcar':
                    fd.write(self.aug)
                if self.is_spin_polarized():
                    if format == 'chg':
                        fd.write('\n')
                    for dim in chg.shape:
                        fd.write(' %4i' % dim)
                    fd.write('\n')  # a new line after dim is required
                    self._write_chg(fd, self.chgdiff[ii], vol, format)
                    if format == 'chgcar':
                        # a new line is always provided self._write_chg
                        fd.write(self.augdiff)
                if format == 'chg' and len(self.chg) > 1:
                    fd.write('\n')


class VaspDos:
    """Class for representing density-of-states produced by VASP

    The energies are in property self.energy

    Site-projected DOS is accesible via the self.site_dos method.

    Total and integrated DOS is accessible as numpy.ndarray's in the
    properties self.dos and self.integrated_dos. If the calculation is
    spin polarized, the arrays will be of shape (2, NDOS), else (1,
    NDOS).

    The self.efermi property contains the currently set Fermi
    level. Changing this value shifts the energies.

    """

    def __init__(self, doscar='DOSCAR', efermi=0.0):
        """Initialize"""
        self._efermi = 0.0
        self.read_doscar(doscar)
        self.efermi = efermi

        # we have determine the resort to correctly map ase atom index to the
        # POSCAR.
        self.sort = []
        self.resort = []
        if os.path.isfile('ase-sort.dat'):
            with open('ase-sort.dat') as file:
                lines = file.readlines()
            for line in lines:
                data = line.split()
                self.sort.append(int(data[0]))
                self.resort.append(int(data[1]))

    def _set_efermi(self, efermi):
        """Set the Fermi level."""
        ef = efermi - self._efermi
        self._efermi = efermi
        self._total_dos[0, :] = self._total_dos[0, :] - ef
        try:
            self._site_dos[:, 0, :] = self._site_dos[:, 0, :] - ef
        except IndexError:
            pass

    def _get_efermi(self):
        return self._efermi

    efermi = property(_get_efermi, _set_efermi, None, "Fermi energy.")

    def _get_energy(self):
        """Return the array with the energies."""
        return self._total_dos[0, :]

    energy = property(_get_energy, None, None, "Array of energies")

    def site_dos(self, atom, orbital):
        """Return an NDOSx1 array with dos for the chosen atom and orbital.

        atom: int
            Atom index
        orbital: int or str
            Which orbital to plot

        If the orbital is given as an integer:
        If spin-unpolarized calculation, no phase factors:
        s = 0, p = 1, d = 2
        Spin-polarized, no phase factors:
        s-up = 0, s-down = 1, p-up = 2, p-down = 3, d-up = 4, d-down = 5
        If phase factors have been calculated, orbitals are
        s, py, pz, px, dxy, dyz, dz2, dxz, dx2
        double in the above fashion if spin polarized.

        """
        # Correct atom index for resorting if we need to. This happens when the
        # ase-sort.dat file exists, and self.resort is not empty.
        if self.resort:
            atom = self.resort[atom]

        # Integer indexing for orbitals starts from 1 in the _site_dos array
        # since the 0th column contains the energies
        if isinstance(orbital, int):
            return self._site_dos[atom, orbital + 1, :]
        n = self._site_dos.shape[1]

        from .vasp_data import PDOS_orbital_names_and_DOSCAR_column
        norb = PDOS_orbital_names_and_DOSCAR_column[n]

        return self._site_dos[atom, norb[orbital.lower()], :]

    def _get_dos(self):
        if self._total_dos.shape[0] == 3:
            return self._total_dos[1, :]
        elif self._total_dos.shape[0] == 5:
            return self._total_dos[1:3, :]

    dos = property(_get_dos, None, None, 'Average DOS in cell')

    def _get_integrated_dos(self):
        if self._total_dos.shape[0] == 3:
            return self._total_dos[2, :]
        elif self._total_dos.shape[0] == 5:
            return self._total_dos[3:5, :]

    integrated_dos = property(_get_integrated_dos, None, None,
                              'Integrated average DOS in cell')

    def read_doscar(self, fname="DOSCAR"):
        """Read a VASP DOSCAR file"""
        with open(fname) as fd:
            natoms = int(fd.readline().split()[0])
            [fd.readline() for _ in range(4)]
            # First we have a block with total and total integrated DOS
            ndos = int(fd.readline().split()[2])
            dos = []
            for _ in range(ndos):
                dos.append(np.array([float(x) for x in fd.readline().split()]))
            self._total_dos = np.array(dos).T
            # Next we have one block per atom, if INCAR contains the stuff
            # necessary for generating site-projected DOS
            dos = []
            for _ in range(natoms):
                line = fd.readline()
                if line == '':
                    # No site-projected DOS
                    break
                ndos = int(line.split()[2])
                line = fd.readline().split()
                cdos = np.empty((ndos, len(line)))
                cdos[0] = np.array(line)
                for nd in range(1, ndos):
                    line = fd.readline().split()
                    cdos[nd] = np.array([float(x) for x in line])
                dos.append(cdos.T)
            self._site_dos = np.array(dos)
