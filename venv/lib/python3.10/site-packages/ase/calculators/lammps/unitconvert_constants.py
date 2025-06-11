# fmt: off

# The following definitions are all given in SI and are excerpted from the
# kim_units.cpp file created by Prof. Ellad B. Tadmor (UMinn) distributed with
# LAMMPS. Note that these values do not correspond to any official CODATA set
# already released, but rather to the values used internally by LAMMPS.
#
# Source: https://physics.nist.gov/cuu/Constants/Table/allascii.txt

from numpy import sqrt

# Constants
boltz_si = 1.38064852e-23  # [J K^-1] Boltzmann's factor (NIST value)
Nav = mole_si = 6.022140857e23  # [unitless] Avogadro's number
me_si = 9.10938356e-31  # [kg] electron rest mass
e_si = 1.6021766208e-19  # [C] elementary charge

# Distance units
meter_si = 1.0
bohr_si = (
    5.2917721067e-11
)  # [m] Bohr unit (distance between nucleus and electron in H)
angstrom_si = 1e-10  # [m] Angstrom
centimeter_si = 1e-2  # [m] centimeter
micrometer_si = 1e-6  # [m] micrometer (micron)
nanometer_si = 1e-9  # [m] nanometer

# Mass units
kilogram_si = 1.0
amu_si = gram_per_mole_si = 1e-3 / Nav  # [kg] gram per mole i.e. amu
gram_si = 1e-3  # [kg] gram
picogram_si = 1e-15  # [kg] picogram
attogram_si = 1e-21  # [kg[ attogram

# Time units
second_si = 1.0
atu_si = 2.418884326509e-17  # [s] atomic time unit
# ( = hbar/E_h where E_h is the  Hartree energy) (NIST value)
atu_electron_si = atu_si * sqrt(
    amu_si / me_si
)  # [s] atomic time unit used in electron system
microsecond_si = 1e-6  # [s] microsecond
nanosecond_si = 1e-9  # [s] nanosecond
picosecond_si = 1e-12  # [s] picosecond
femtosecond_si = 1e-15  # [s] femtosecond

# Density units
gram_per_centimetercu_si = (
    gram_si / centimeter_si ** 3
)  # [kg/m^3] gram/centimeter^3
amu_per_bohrcu_si = amu_si / bohr_si ** 3  # [kg/m^3] amu/bohr^3
picogram_per_micrometercu_si = (
    picogram_si / micrometer_si ** 3
)  # [kg/m^3] picogram/micrometer^3
attogram_per_nanometercu_si = (
    attogram_si / nanometer_si ** 3
)  # [kg/m^3] attogram/nanometer^3

# Energy/torque units
joule_si = 1.0
kcal_si = (
    4184.0
)  # [J] kilocalorie (heat energy involved in warming up one kilogram of
# water by one degree Kelvin)
ev_si = (
    1.6021766208e-19
)  # [J] electron volt (amount of energy gained or lost by the
# charge of a single electron moving across an electric
# potential difference of one volt.) (NIST value)
hartree_si = (
    4.359744650e-18
)  # [J] Hartree (approximately the electric potential energy
# of the hydrogen atom in its ground state) (NIST value)
kcal_per_mole_si = kcal_si / Nav  # [J] kcal/mole
erg_si = 1e-7  # [J] erg
dyne_centimeter_si = 1e-7  # [J[ dyne*centimeter
picogram_micrometersq_per_microsecondsq_si = (
    picogram_si * micrometer_si ** 2 / microsecond_si ** 2
)  # [J] picogram*micrometer^2/microsecond^2
attogram_nanometersq_per_nanosecondsq_si = (
    attogram_si * nanometer_si ** 2 / nanosecond_si ** 2
)  # [J] attogram*nanometer^2/nanosecond^2

# Velocity units
meter_per_second_si = 1.0
angstrom_per_femtosecond_si = (
    angstrom_si / femtosecond_si
)  # [m/s] Angstrom/femtosecond
angstrom_per_picosecond_si = (
    angstrom_si / picosecond_si
)  # [m/s] Angstrom/picosecond
micrometer_per_microsecond_si = (
    micrometer_si / microsecond_si
)  # [m/s] micrometer/microsecond
nanometer_per_nanosecond_si = (
    nanometer_si / nanosecond_si
)  # [m/s] nanometer/nanosecond
centimeter_per_second_si = centimeter_si  # [m/s] centimeter/second
bohr_per_atu_si = bohr_si / atu_electron_si  # [m/s] bohr/atu

# Force units
newton_si = 1.0
kcal_per_mole_angstrom_si = (
    kcal_per_mole_si / angstrom_si
)  # [N] kcal/(mole*Angstrom)
ev_per_angstrom_si = ev_si / angstrom_si  # [N] eV/Angstrom
dyne_si = dyne_centimeter_si / centimeter_si  # [N] dyne
hartree_per_bohr_si = hartree_si / bohr_si  # [N] hartree/bohr
picogram_micrometer_per_microsecondsq_si = (
    picogram_si * micrometer_si / microsecond_si ** 2
)  # [N] picogram*micrometer/microsecond^2
attogram_nanometer_per_nanosecondsq_si = (
    attogram_si * nanometer_si / nanosecond_si ** 2
)  # [N] attogram*nanometer/nanosecond^2

# Temperature units
kelvin_si = 1.0

# Pressure units
pascal_si = 1.0
atmosphere_si = 101325.0  # [Pa] standard atmosphere (NIST value)
bar_si = 1e5  # [Pa] bar
dyne_per_centimetersq_si = (
    dyne_centimeter_si / centimeter_si ** 3
)  # [Pa] dyne/centimeter^2
picogram_per_micrometer_microsecondsq_si = picogram_si / (
    micrometer_si * microsecond_si ** 2
)  # [Pa] picogram/(micrometer*microsecond^2)
attogram_per_nanometer_nanosecondsq_si = attogram_si / (
    nanometer_si * nanosecond_si ** 2
)  # [Pa] attogram/(nanometer*nanosecond^2)

# Viscosity units
poise_si = 0.1  # [Pa*s] Poise
amu_per_bohr_femtosecond_si = amu_si / (
    bohr_si * femtosecond_si
)  # [Pa*s] amu/(bohr*femtosecond)
picogram_per_micrometer_microsecond_si = picogram_si / (
    micrometer_si * microsecond_si
)  # [Pa*s] picogram/(micrometer*microsecond)
attogram_per_nanometer_nanosecond_si = attogram_si / (
    nanometer_si * nanosecond_si
)  # [Pa*s] attogram/(nanometer*nanosecond)

# Charge units
coulomb_si = 1.0
echarge_si = e_si  # [C] electron charge unit
statcoulomb_si = (
    e_si / 4.8032044e-10
)  # [C] Statcoulomb or esu (value from LAMMPS units documentation)
picocoulomb_si = 1e-12  # [C] picocoulomb

# Dipole units
coulomb_meter_si = 1
electron_angstrom_si = echarge_si * angstrom_si  # [C*m] electron*angstrom
statcoulomb_centimeter_si = (
    statcoulomb_si * centimeter_si
)  # [C*m] statcoulomb*centimeter
debye_si = 1e-18 * statcoulomb_centimeter_si  # [C*m] Debye
picocoulomb_micrometer_si = (
    picocoulomb_si * micrometer_si
)  # [C*m] picocoulomb*micrometer
electron_nanometer_si = echarge_si * nanometer_si  # [C*m] electron*nanometer

# Electric field units
volt_si = 1.0
volt_per_meter_si = 1
volt_per_angstrom_si = 1.0 / angstrom_si  # [V/m] volt/angstrom
statvolt_per_centimeter_si = erg_si / (
    statcoulomb_si * centimeter_si
)  # [V/m] statvolt/centimeter
volt_per_centimeter_si = 1.0 / centimeter_si  # [V/m] volt/centimeter
volt_per_micrometer_si = 1.0 / micrometer_si  # [V/m] volt/micrometer
volt_per_nanometer_si = 1.0 / nanometer_si  # [V/m] volt/nanometer
