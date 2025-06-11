import numpy as np
import pytest

from ase.quaternions import Quaternion

TEST_N = 200


def axang_rotm(u, theta):

    u = np.array(u, float)
    u /= np.linalg.norm(u)

    # Cross product matrix for u
    ucpm = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

    # Rotation matrix
    rotm = (np.cos(theta) * np.identity(3) + np.sin(theta) * ucpm +
            (1 - np.cos(theta)) * np.kron(u[:, None], u[None, :]))

    return rotm


def rand_rotm(rng=np.random.RandomState(0)):
    """Axis & angle rotations."""
    u = rng.random(3)
    theta = rng.random() * np.pi * 2

    return axang_rotm(u, theta)


def eulang_rotm(a, b, c, mode='zyz'):

    rota = axang_rotm([0, 0, 1], a)
    rotc = axang_rotm([0, 0, 1], c)

    if mode == 'zyz':
        rotb = axang_rotm([0, 1, 0], b)
    elif mode == 'zxz':
        rotb = axang_rotm([1, 0, 0], b)

    return np.dot(rotc, np.dot(rotb, rota))


@pytest.fixture()
def rng():
    return np.random.RandomState(0)


def test_quaternions_rotations(rng):

    # First: test that rotations DO work
    for _ in range(TEST_N):
        # n random tests

        rotm = rand_rotm(rng)

        q = Quaternion.from_matrix(rotm)
        assert np.allclose(rotm, q.rotation_matrix())

        # Now test this with a vector
        v = rng.random(3)

        vrotM = np.dot(rotm, v)
        vrotQ = q.rotate(v)

        assert np.allclose(vrotM, vrotQ)


def test_quaternions_gimbal(rng):

    # Second: test the special case of a PI rotation

    rotm = np.identity(3)
    rotm[:2, :2] *= -1               # Rotate PI around z axis

    q = Quaternion.from_matrix(rotm)

    assert not np.isnan(q.q).any()


def test_quaternions_overload(rng):

    # Third: test compound rotations and operator overload
    for _ in range(TEST_N):
        rotm1 = rand_rotm(rng)
        rotm2 = rand_rotm(rng)

        q1 = Quaternion.from_matrix(rotm1)
        q2 = Quaternion.from_matrix(rotm2)

        assert np.allclose(np.dot(rotm2, rotm1),
                           (q2 * q1).rotation_matrix())
        # Now test this with a vector
        v = rng.random(3)

        vrotM = np.dot(rotm2, np.dot(rotm1, v))
        vrotQ = (q2 * q1).rotate(v)

        assert np.allclose(vrotM, vrotQ)


@pytest.mark.parametrize('mode', ['zyz', 'zxz'])
def test_quaternions_euler(rng: np.random.RandomState, mode: str):
    # Fourth: test Euler angles
    for _ in range(TEST_N):
        a, c = rng.uniform(-np.pi, np.pi, size=2)
        b = rng.uniform(0, np.pi)

        q_eul = Quaternion.from_euler_angles(a, b, c, mode=mode)
        rot_eul = eulang_rotm(a, b, c, mode=mode)
        assert np.allclose(rot_eul, q_eul.rotation_matrix())

        # Test conversion back and forth
        a2, b2, c2 = q_eul.euler_angles(mode=mode)
        assert np.allclose([a2, b2, c2], [a, b, c])
        q_eul_2 = Quaternion.from_euler_angles(a2, b2, c2, mode=mode)
        assert np.allclose(q_eul_2.q, q_eul.q)


def test_quaternions_rotm(rng):

    # Fifth: test that conversion back to rotation matrices works properly
    for _ in range(TEST_N):
        rotm1 = rand_rotm(rng)
        rotm2 = rand_rotm(rng)

        q1 = Quaternion.from_matrix(rotm1)
        q2 = Quaternion.from_matrix(rotm2)

        assert np.allclose(q1.rotation_matrix(), rotm1)
        assert np.allclose(q2.rotation_matrix(), rotm2)
        assert np.allclose((q1 * q2).rotation_matrix(), np.dot(rotm1, rotm2))
        assert np.allclose((q1 * q2).rotation_matrix(), np.dot(rotm1, rotm2))


def test_quaternions_axang(rng):

    # Sixth: test conversion to axis + angle
    q = Quaternion()
    n, theta = q.axis_angle()
    assert theta == 0

    u = np.array([1, 0.5, 1])
    u /= np.linalg.norm(u)
    alpha = 1.25

    q = Quaternion.from_matrix(axang_rotm(u, alpha))
    n, theta = q.axis_angle()

    assert np.isclose(theta, alpha)
    assert np.allclose(u, n)


@pytest.mark.parametrize(
    'q,euler_angles,mode,rotation_matrix,axis,angle', [
        # pi/2 rotation along z axis
        (
            np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]),
            np.array([0, 0, np.pi / 2]),  # We use alpha=0 for singular case
            'zyz',
            np.array([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ]),
            np.array([0, 0, 1]),
            np.pi / 2,
        ),
        # pi/2 rotation along x axis
        (
            np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]),
            np.array([np.pi / 2, np.pi / 2, -np.pi / 2]),
            'zyz',
            np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ]),
            np.array([1, 0, 0]),
            np.pi / 2,
        ),
    ]
)
def test_quaternions_special_cases(
    q, euler_angles, mode, rotation_matrix, axis, angle
):
    quaternions = [
        Quaternion(q),
        Quaternion.from_euler_angles(*euler_angles, mode=mode),
        Quaternion.from_matrix(rotation_matrix),
        Quaternion.from_axis_angle(axis, angle),
    ]
    for quaternion in quaternions:
        axis_actual, angle_actual = quaternion.axis_angle()

        assert np.allclose(quaternion.q, q)
        assert np.allclose(quaternion.euler_angles(), euler_angles)
        assert np.allclose(quaternion.rotation_matrix(), rotation_matrix)
        assert np.allclose(axis_actual, axis)
        assert np.isclose(angle_actual, angle)
