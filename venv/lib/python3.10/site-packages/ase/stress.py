# fmt: off

import numpy as np

# The indices of the full stiffness matrix of (orthorhombic) interest
voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def get_elasticity_tensor(atoms, h=0.001, verbose=False):
    """

             1    dE         dσ_ij
    C     =  - ----------- = -----
     ijkl    V dε_ij dε_kl   dε_kl

    """
    cell0 = atoms.cell.copy()
    C_ijkl = np.zeros((3, 3, 3, 3))
    f = voigt_6_to_full_3x3_stress
    for k in range(3):
        for l in range(3):
            strain = np.eye(3)
            strain[k, l] += h
            atoms.set_cell(cell0 @ strain, scale_atoms=True)
            stressp_ij = f(atoms.get_stress())
            strain[k, l] -= 2 * h
            atoms.set_cell(cell0 @ strain, scale_atoms=True)
            stressm_ij = f(atoms.get_stress())
            C_ijkl[k, l] = (stressp_ij - stressm_ij) / (2 * h)

    if verbose:
        for i in range(3):
            for j in range(3):
                print(f'C_ijkl[{i}, {j}] =')
                for k in range(3):
                    for l in range(3):
                        print(round(C_ijkl[i, j, k, l], 2), end=' ')
                    print()
                print()
            print()

    return C_ijkl


def full_3x3_to_voigt_6_index(i, j):
    if i == j:
        return i
    return 6 - i - j


def voigt_6_to_full_3x3_strain(strain_vector):
    """
    Form a 3x3 strain matrix from a 6 component vector in Voigt notation
    """
    e1, e2, e3, e4, e5, e6 = np.transpose(strain_vector)
    return np.transpose([[1.0 + e1, 0.5 * e6, 0.5 * e5],
                         [0.5 * e6, 1.0 + e2, 0.5 * e4],
                         [0.5 * e5, 0.5 * e4, 1.0 + e3]])


def voigt_6_to_full_3x3_stress(stress_vector):
    """
    Form a 3x3 stress matrix from a 6 component vector in Voigt notation
    """
    s1, s2, s3, s4, s5, s6 = np.transpose(stress_vector)
    return np.transpose([[s1, s6, s5],
                         [s6, s2, s4],
                         [s5, s4, s3]])


def full_3x3_to_voigt_6_strain(strain_matrix):
    """
    Form a 6 component strain vector in Voigt notation from a 3x3 matrix
    """
    strain_matrix = np.asarray(strain_matrix)
    return np.transpose([strain_matrix[..., 0, 0] - 1.0,
                         strain_matrix[..., 1, 1] - 1.0,
                         strain_matrix[..., 2, 2] - 1.0,
                         strain_matrix[..., 1, 2] + strain_matrix[..., 2, 1],
                         strain_matrix[..., 0, 2] + strain_matrix[..., 2, 0],
                         strain_matrix[..., 0, 1] + strain_matrix[..., 1, 0]])


def full_3x3_to_voigt_6_stress(stress_matrix):
    """
    Form a 6 component stress vector in Voigt notation from a 3x3 matrix
    """
    stress_matrix = np.asarray(stress_matrix)
    return np.transpose([stress_matrix[..., 0, 0],
                         stress_matrix[..., 1, 1],
                         stress_matrix[..., 2, 2],
                         (stress_matrix[..., 1, 2] +
                          stress_matrix[..., 2, 1]) / 2,
                         (stress_matrix[..., 0, 2] +
                          stress_matrix[..., 2, 0]) / 2,
                         (stress_matrix[..., 0, 1] +
                          stress_matrix[..., 1, 0]) / 2])
