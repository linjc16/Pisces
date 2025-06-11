# fmt: off

import numpy as np

from ase.atoms import Atoms


class Quaternions(Atoms):

    def __init__(self, *args, **kwargs):
        quaternions = None
        if 'quaternions' in kwargs:
            quaternions = np.array(kwargs['quaternions'])
            del kwargs['quaternions']
        Atoms.__init__(self, *args, **kwargs)
        if quaternions is not None:
            self.set_array('quaternions', quaternions, shape=(4,))
            # set default shapes
            self.set_shapes(np.array([[3, 2, 1]] * len(self)))

    def set_shapes(self, shapes):
        self.set_array('shapes', shapes, shape=(3,))

    def set_quaternions(self, quaternions):
        self.set_array('quaternions', quaternions, quaternion=(4,))

    def get_shapes(self):
        return self.get_array('shapes')

    def get_quaternions(self):
        return self.get_array('quaternions').copy()


class Quaternion:

    def __init__(self, qin=[1, 0, 0, 0]):
        assert len(qin) == 4
        self.q = np.array(qin)

    def __str__(self):
        return self.q.__str__()

    def __mul__(self, other):
        sw, sx, sy, sz = self.q
        ow, ox, oy, oz = other.q
        return Quaternion([sw * ow - sx * ox - sy * oy - sz * oz,
                           sw * ox + sx * ow + sy * oz - sz * oy,
                           sw * oy + sy * ow + sz * ox - sx * oz,
                           sw * oz + sz * ow + sx * oy - sy * ox])

    def conjugate(self):
        return Quaternion(-self.q * np.array([-1., 1., 1., 1.]))

    def rotate(self, vector):
        """Apply the rotation matrix to a vector."""
        qw, qx, qy, qz = self.q[0], self.q[1], self.q[2], self.q[3]
        x, y, z = vector[0], vector[1], vector[2]

        ww = qw * qw
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz

        return np.array(
            [(ww + xx - yy - zz) * x + 2 * ((xy - wz) * y + (xz + wy) * z),
             (ww - xx + yy - zz) * y + 2 * ((xy + wz) * x + (yz - wx) * z),
             (ww - xx - yy + zz) * z + 2 * ((xz - wy) * x + (yz + wx) * y)])

    def rotation_matrix(self):

        qw, qx, qy, qz = self.q[0], self.q[1], self.q[2], self.q[3]

        ww = qw * qw
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz

        return np.array([[ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
                         [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
                         [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz]])

    def axis_angle(self):
        """Returns axis and angle (in radians) for the rotation described
        by this Quaternion"""

        sinth_2 = np.linalg.norm(self.q[1:])

        if sinth_2 == 0:
            # The angle is zero
            theta = 0.0
            n = np.array([0, 0, 1])
        else:
            theta = np.arctan2(sinth_2, self.q[0]) * 2
            n = self.q[1:] / sinth_2

        return n, theta

    def euler_angles(self, mode='zyz'):
        """Return three Euler angles describing the rotation, in radians.
        Mode can be zyz or zxz. Default is zyz."""
        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0276302
        if mode == 'zyz':
            a, b, c, d = self.q[0], self.q[3], self.q[2], -self.q[1]
        elif mode == 'zxz':
            a, b, c, d = self.q[0], self.q[3], self.q[1], self.q[2]
        else:
            raise ValueError(f'Invalid Euler angles mode {mode}')

        beta = 2 * np.arccos(
            np.sqrt((a**2 + b**2) / (a**2 + b**2 + c**2 + d**2))
        )
        gap = np.arctan2(b, a)  # (gamma + alpha) / 2
        gam = np.arctan2(d, c)  # (gamma - alpha) / 2
        if np.isclose(beta, 0):
            # gam is meaningless here
            alpha = 0
            gamma = 2 * gap - alpha
        elif np.isclose(beta, np.pi):
            # gap is meaningless here
            alpha = 0
            gamma = 2 * gam + alpha
        else:
            alpha = gap - gam
            gamma = gap + gam

        return np.array([alpha, beta, gamma])

    def arc_distance(self, other):
        """Gives a metric of the distance between two quaternions,
        expressed as 1-|q1.q2|"""

        return 1.0 - np.abs(np.dot(self.q, other.q))

    @staticmethod
    def rotate_byq(q, vector):
        """Apply the rotation matrix to a vector."""
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        x, y, z = vector[0], vector[1], vector[2]

        ww = qw * qw
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz

        return np.array(
            [(ww + xx - yy - zz) * x + 2 * ((xy - wz) * y + (xz + wy) * z),
             (ww - xx + yy - zz) * y + 2 * ((xy + wz) * x + (yz - wx) * z),
             (ww - xx - yy + zz) * z + 2 * ((xz - wy) * x + (yz + wx) * y)])

    @staticmethod
    def from_matrix(matrix):
        """Build quaternion from rotation matrix."""
        m = np.array(matrix)
        assert m.shape == (3, 3)

        # Now we need to find out the whole quaternion
        # This method takes into account the possibility of qw being nearly
        # zero, so it picks the stablest solution

        if m[2, 2] < 0:
            if (m[0, 0] > m[1, 1]):
                # Use x-form
                qx = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) / 2.0
                fac = 1.0 / (4 * qx)
                qw = (m[2, 1] - m[1, 2]) * fac
                qy = (m[0, 1] + m[1, 0]) * fac
                qz = (m[0, 2] + m[2, 0]) * fac
            else:
                # Use y-form
                qy = np.sqrt(1 - m[0, 0] + m[1, 1] - m[2, 2]) / 2.0
                fac = 1.0 / (4 * qy)
                qw = (m[0, 2] - m[2, 0]) * fac
                qx = (m[0, 1] + m[1, 0]) * fac
                qz = (m[1, 2] + m[2, 1]) * fac
        else:
            if (m[0, 0] < -m[1, 1]):
                # Use z-form
                qz = np.sqrt(1 - m[0, 0] - m[1, 1] + m[2, 2]) / 2.0
                fac = 1.0 / (4 * qz)
                qw = (m[1, 0] - m[0, 1]) * fac
                qx = (m[2, 0] + m[0, 2]) * fac
                qy = (m[1, 2] + m[2, 1]) * fac
            else:
                # Use w-form
                qw = np.sqrt(1 + m[0, 0] + m[1, 1] + m[2, 2]) / 2.0
                fac = 1.0 / (4 * qw)
                qx = (m[2, 1] - m[1, 2]) * fac
                qy = (m[0, 2] - m[2, 0]) * fac
                qz = (m[1, 0] - m[0, 1]) * fac

        return Quaternion(np.array([qw, qx, qy, qz]))

    @staticmethod
    def from_axis_angle(n, theta):
        """Build quaternion from axis (n, vector of 3 components) and angle
        (theta, in radianses)."""

        n = np.array(n, float) / np.linalg.norm(n)
        return Quaternion(np.concatenate([[np.cos(theta / 2.0)],
                                          np.sin(theta / 2.0) * n]))

    @staticmethod
    def from_euler_angles(a, b, c, mode='zyz'):
        """Build quaternion from Euler angles, given in radians. Default
        mode is ZYZ, but it can be set to ZXZ as well."""

        q_a = Quaternion.from_axis_angle([0, 0, 1], a)
        q_c = Quaternion.from_axis_angle([0, 0, 1], c)

        if mode == 'zyz':
            q_b = Quaternion.from_axis_angle([0, 1, 0], b)
        elif mode == 'zxz':
            q_b = Quaternion.from_axis_angle([1, 0, 0], b)
        else:
            raise ValueError(f'Invalid Euler angles mode {mode}')

        return q_c * q_b * q_a
