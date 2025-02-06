import numpy as np

class Kinematics:
    def __init__(self):
        self.rad2deg = 180 / np.pi
        self.deg2rad = np.pi / 180

    @staticmethod
    def cross_matrix(w):
        """
        Skew-symmetric matrix for cross product.
        :param w: 1x3 numpy array (angular velocity vector)
        :return: 3x3 skew-symmetric numpy matrix
        """
        return np.array([[0, -w[2], w[1]],
                         [w[2], 0, -w[0]],
                         [-w[1], w[0], 0]])
    
    @staticmethod
    def rotmax(angle):
        """
        Rotation matrix for x-axis rotation (DCM).
        :param angle: Angle in radians
        :return: 3x3 rotation matrix
        """
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0],
                         [0, cosine, sine],
                         [0, -sine, cosine]])

    @staticmethod
    def rotmay(angle):
        """
        Rotation matrix for y-axis rotation (DCM).
        :param angle: Angle in radians
        :return: 3x3 rotation matrix
        """
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array([[cosine, 0, -sine],
                         [0, 1, 0],
                         [sine, 0, cosine]])

    @staticmethod
    def rotmaz(angle):
        """
        Rotation matrix for z-axis rotation (DCM).
        :param angle: Angle in radians
        :return: 3x3 rotation matrix
        """
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array([[cosine, sine, 0],
                         [-sine, cosine, 0],
                         [0, 0, 1]])

    @staticmethod
    def rotmax_rx(angle):
        """
        Rotation matrix for x-axis rotation.
        :param angle: Angle in radians
        :return: 3x3 rotation matrix
        """
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0],
                         [0, cosine, -sine],
                         [0, sine, cosine]])

    @staticmethod
    def rotmay_rx(angle):
        """
        Rotation matrix for y-axis rotation.
        :param angle: Angle in radians
        :return: 3x3 rotation matrix
        """
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array([[cosine, 0, sine],
                         [0, 1, 0],
                         [-sine, 0, cosine]])

    @staticmethod
    def rotmaz_rx(angle):
        """
        Rotation matrix for z-axis rotation.
        :param angle: Angle in radians
        :return: 3x3 rotation matrix
        """
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array([[cosine, -sine, 0],
                         [sine, cosine, 0],
                         [0, 0, 1]])
    
    def eulerrmx(self, euler_angle, euler_vector):
        """
        Rodrigues' rotation formula.
        :param euler_angle: Rotation angle in radians
        :param euler_vector: 1x3 numpy array (rotation axis)
        :return: 3x3 rotation matrix
        """
        cosine_0 = np.cos(euler_angle)
        sine = np.sin(euler_angle)
        cosine_1 = 1 - cosine_0
        euler_vector = np.reshape(euler_vector, (3, 1))

        rot_mat = (cosine_0 * np.eye(3) + 
                   cosine_1 * np.dot(euler_vector, euler_vector.T) + 
                   sine * Kinematics.cross_matrix(euler_vector.flatten()))

        return rot_mat

    @staticmethod
    def rmxeuler(rot_mat):
        """
        abstract Rodrigues rotation elements
        input
        rot_mat : 3*3 radian angle matrix
        output
        euler_angle  : scalar radian angle
        euler_vector : 1*3 numpy array
        """
        trace = np.trace(rot_mat)
        if trace == 3:
            euler_angle = 0
            euler_vector = np.array([1, 0, 0])
        elif trace < -0.99999:
            euler_angle = np.pi
            w = np.diagonal(rot_mat)
            euler_vector = np.sqrt((1 + w) / 2)
            if euler_vector[0] > 0.5:
                euler_vector[1] = np.sign(rot_mat[0, 1]) * euler_vector[1]
                euler_vector[2] = np.sign(rot_mat[2, 0]) * euler_vector[2]
            elif euler_vector[1] > 0.5:
                euler_vector[0] = np.sign(rot_mat[0, 1]) * euler_vector[0]
                euler_vector[2] = np.sign(rot_mat[1, 2]) * euler_vector[2]
            else:
                euler_vector[0] = np.sign(rot_mat[2, 0]) * euler_vector[0]
                euler_vector[1] = np.sign(rot_mat[1, 2]) * euler_vector[1]
        else:
            euler_angle = np.arccos((trace - 1) / 2)
            sine = np.sin(euler_angle)
            euler_vector = np.array([
                rot_mat[1, 2] - rot_mat[2, 1],
                rot_mat[2, 0] - rot_mat[0, 2],
                rot_mat[0, 1] - rot_mat[1, 0]
            ]) / (2 * sine)

        return euler_angle, euler_vector
    
    @staticmethod
    def rmxexyz(rot_mat):
        """
        abstract X-Y-Z rotation angles from rotation matrix
        input
        rot_mat : 3*3 numpy array
        output
        euler_angles : 1*3 numpy array
        """
        a11, a12, a21, a22, a31, a32, a33 = rot_mat[0, 0], rot_mat[0, 1], rot_mat[1, 0], rot_mat[1, 1], rot_mat[2, 0], \
        rot_mat[2, 1], rot_mat[2, 2]

        if abs(a31) <= 1:
            eul2 = np.arcsin(a31)
        elif a31 < 0:
            eul2 = -np.pi / 2
        else:
            eul2 = np.pi / 2

        if abs(a31) <= 0.99999:
            if a33 != 0:
                eul1 = np.arctan2(-a32, a33)
                if eul1 > np.pi:
                    eul1 = eul1 - 2 * np.pi
            else:
                eul1 = np.pi / 2 * np.sign(-a32)

            if a11 != 0:
                eul3 = np.arctan2(-a21, a11)
                if eul3 > np.pi:
                    eul3 = eul3 - 2 * np.pi
            else:
                eul3 = np.pi / 2 * np.sign(-a21)
        else:
            eul1 = 0
            if a22 != 0:
                eul3 = np.arctan2(a12, a22)
                if eul3 > np.pi:
                    eul3 = eul3 - 2 * np.pi
            else:
                eul3 = np.pi / 2 * np.sign(a12)

        euler_angles = np.array([eul1, eul2, eul3])

        return euler_angles

    @staticmethod
    def rmxezxy(rot_mat):
        """
        abstract Z-X-Y rotation angles from rotation matrix
        input
        rot_mat : 3*3 numpy array
        output
        euler_angles : 1*3 numpy array
        """
        spct = -rot_mat[0, 2]
        ctsf = -rot_mat[1, 0]
        ctcf = rot_mat[1, 1]
        stet = rot_mat[1, 2]
        cpct = rot_mat[2, 2]

        if abs(stet) <= 1:
            eul2 = np.arcsin(stet)
        else:
            eul2 = np.pi / 2 * np.sign(stet)

        if abs(eul2) <= np.pi / 2 - 1e-5:
            if abs(ctcf) != 0:
                eul1 = np.arctan2(ctsf, ctcf)
                if eul1 > np.pi:
                    eul1 = eul1 - 2 * np.pi
            else:
                eul1 = np.pi / 2 * np.sign(ctsf)

            if abs(cpct) != 0:
                eul3 = np.arctan2(spct, cpct)
                if eul3 > np.pi:
                    eul3 = eul3 - 2 * np.pi
            else:
                eul3 = np.pi / 2 * np.sign(spct)
        else:
            capb = rot_mat[0, 0]
            sapb = rot_mat[0, 1]
            eul1 = 0.
            if abs(capb) != 0:
                eul3 = np.arctan2(sapb, capb)
                if eul3 > np.pi:
                    eul3 = eul3 - 2 * np.pi
            else:
                eul3 = 0.

        euler_angles = np.array([eul1, eul2, eul3])

        return euler_angles

    @staticmethod
    def rmxezxz(rot_mat):
        """
        abstract Z-X-Z rotation angles from rotation matrix
        input
        rot_mat : 3*3 numpy array
        output
        euler_angles : 1*3 numpy array
        """
        a11, a12, a13, a23, a31, a32, a33 = rot_mat[0, 0], rot_mat[0, 1], rot_mat[0, 2], rot_mat[1, 2], rot_mat[2, 0], \
        rot_mat[2, 1], rot_mat[2, 2]

        if abs(a33) <= 1:
            eul2 = np.arccos(a33)
        elif a33 < 0:
            eul2 = np.pi
        else:
            eul2 = 0

        if abs(eul2) >= 0.00001:
            if a32 != 0:
                eul1 = np.arctan2(a31, -a32)
            else:
                eul1 = np.pi / 2 * np.sign(a31)
                if eul1 > np.pi:
                    eul1 = eul1 - 2 * np.pi

            if a23 != 0:
                eul3 = np.arctan2(a13, a23)
                if eul3 > np.pi:
                    eul3 = eul3 - 2 * np.pi
            else:
                eul3 = np.pi / 2 * np.sign(a13)
        else:
            eul1 = 0
            if a11 != 0:
                eul3 = np.arctan2(a12, a11)
                if eul3 > np.pi:
                    eul3 = eul3 - 2 * np.pi
            else:
                eul3 = np.pi / 2 * np.sign(a12)

        euler_angles = np.array([eul1, eul2, eul3])

        return euler_angles

    @staticmethod
    def rmxezyx(rot_mat):
        """
        abstract Z-Y-X rotation angles from rotation matrix
        input
        rot_mat : 3*3 numpy array
        output
        euler_angles : 1*3 numpy array
        """
        stet = -rot_mat[0, 2]
        ctsf = rot_mat[0, 1]
        ctcf = rot_mat[0, 0]
        spct = rot_mat[1, 2]
        cpct = rot_mat[2, 2]

        if abs(stet) <= 1.:
            eul2 = np.arcsin(stet)
        else:
            eul2 = np.pi / 2 * np.sign(stet)

        if abs(eul2) <= np.pi / 2 - 1e-5:
            if abs(ctcf) != 0:
                eul1 = np.arctan2(ctsf, ctcf)
                if eul1 > np.pi:
                    eul1 = eul1 - 2 * np.pi
            else:
                eul1 = np.pi / 2 * np.sign(ctsf)

            if abs(cpct) != 0:
                eul3 = np.arctan2(spct, cpct)
                if eul3 > np.pi:
                    eul3 = eul3 - 2 * np.pi
            else:
                eul3 = np.pi / 2 * np.sign(spct)
        else:
            capb = rot_mat[1, 1]
            sapb = rot_mat[1, 0]
            eul1 = 0.

            if abs(capb) != 0:
                eul3 = np.arctan2(sapb, capb)
                if eul3 > np.pi:
                    eul3 = eul3 - 2 * np.pi
            else:
                eul3 = 0.

        euler_angles = np.array([eul1, eul2, eul3])

        return euler_angles

    @staticmethod
    def rmxquat(rot_mat):
        """
        Obtain the attitude quaternions from the attitude rotation matrix.

        Parameters:
        rot_mat : np.array
            Rotation matrix (3x3).

        Returns:
        quaternions : np.array
            Attitude quaternions with the real part as the first element.
        """
        matra = np.trace(rot_mat)
        auxi = 1 - matra
        selec = np.array([1 + matra, auxi + 2 * rot_mat[0, 0], auxi + 2 * rot_mat[1, 1], auxi + 2 * rot_mat[2, 2]])
        ites = np.argmax(selec)
        auxi = 0.5 * np.sqrt(selec[ites])

        if ites == 0:
            quaternions = np.array([
                auxi,
                (rot_mat[1, 2] - rot_mat[2, 1]) / (4 * auxi),
                (rot_mat[2, 0] - rot_mat[0, 2]) / (4 * auxi),
                (rot_mat[0, 1] - rot_mat[1, 0]) / (4 * auxi)
            ])
        elif ites == 1:
            quaternions = np.array([
                (rot_mat[1, 2] - rot_mat[2, 1]) / (4 * auxi),
                auxi,
                (rot_mat[0, 1] + rot_mat[1, 0]) / (4 * auxi),
                (rot_mat[0, 2] + rot_mat[2, 0]) / (4 * auxi)
            ])
        elif ites == 2:
            quaternions = np.array([
                (rot_mat[2, 0] - rot_mat[0, 2]) / (4 * auxi),
                (rot_mat[0, 1] + rot_mat[1, 0]) / (4 * auxi),
                auxi,
                (rot_mat[1, 2] + rot_mat[2, 1]) / (4 * auxi)
            ])
        else:  # ites == 3
            quaternions = np.array([
                (rot_mat[0, 1] - rot_mat[1, 0]) / (4 * auxi),
                (rot_mat[0, 2] + rot_mat[2, 0]) / (4 * auxi),
                (rot_mat[1, 2] + rot_mat[2, 1]) / (4 * auxi),
                auxi
            ])

        return quaternions

    @staticmethod
    def quat_matrix(q):
        """
        q : quaternion [q_real, q_vec]^T
        Return : 4x4 quaternion matrix
        """
        q_mat = np.array([
            [q[0], -q[1], -q[2], -q[3]],
            [q[1], q[0], -q[3], q[2]],
            [q[2], q[3], q[0], -q[1]],
            [q[3], -q[2], q[1], q[0]]
        ])
        return q_mat

    @staticmethod
    def quat_inv(q):
        """
        q : quaternion [q_real, q_vec]^T
        Return : q_conj = [q_real, -q_vec]^T
        """
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        return q_conj

    @staticmethod
    def quat_norm(q):
        """
        Normalize the quaternion to have a unit norm.

        q : np.array
            Input quaternion [q_real, qx, qy, qz]

        Returns:
        np.array
            Normalized quaternion [q_real, qx, qy, qz]
        """
        norm = np.linalg.norm(q)
        if norm == 0:
            raise ValueError("The quaternion has zero norm and cannot be normalized.")
        q_norm = q / norm
        return q_norm

    @staticmethod
    def quat_prod(quat1, quat2):
        """
        Compute the product of two quaternions.

        quat1 : np.array
            First quaternion [q_real, qx, qy, qz] where Q = q_real + qx i + qy j + qz k.
        quat2 : np.array
            Second quaternion [p_real, px, py, pz] where P = p_real + px i + py j + pz k.

        Returns:
        np.array
            Quaternion product [r_real, rx, ry, rz] where R = Q X P.
        """
        q_real = quat1[0]
        p_real = quat2[0]
        v1 = quat1[1:]
        v2 = quat2[1:]

        # Quaternion product formula
        real_part = q_real * p_real - np.dot(v1, v2)
        imaginary_part = q_real * v2 + p_real * v1 + np.cross(v1, v2)
        quat = np.concatenate(([real_part], imaginary_part))

        return quat

    @staticmethod
    def quat_unity(q):
        """
        q : np.array
            Input quaternion [qw, qx, qy, qz]
        Returns:
            np.array
            Normalized quaternion [qw, qx, qy, qz], such that the square of its norm equals 1.
        """
        qnorm = np.sqrt(np.dot(q, q))
        if (qnorm != 0):
            q_unit = q / qnorm
        else:
            q_unit = np.array([1, 0, 0, 0])

        return q_unit

    @staticmethod
    def quatrmx(quaternion):
        """
        Compute the rotation matrix from the quaternion.

        Parameters:
        quaternion : np.array
            Attitude quaternion [q_real, qx, qy, qz] where Q = q_real + qx i + qy j + qz k.

        Returns:
        rot_mat : np.array
            Rotation matrix (3, 3) as euler rotation matrix (DCM)
        """
        q0, q1, q2, q3 = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        q0q, q1q, q2q, q3q = q0 ** 2, q1 ** 2, q2 ** 2, q3 ** 2
        q01, q02, q03 = 2 * q0 * q1, 2 * q0 * q2, 2 * q0 * q3
        q12, q13, q23 = 2 * q1 * q2, 2 * q1 * q3, 2 * q2 * q3

        rot_mat = np.array([
            [q0q + q1q - q2q - q3q, q12 - q03, q13 + q02],
            [q12 + q03, q0q - q1q + q2q - q3q, q23 - q01],
            [q13 - q02, q23 + q01, q0q - q1q - q2q + q3q]
        ])
        return rot_mat

    def quatexyz(q):
        """
        Parameters:
        quaternion : np.array
            Attitude quaternion [q1, q2, q3, q4] where Q = q1 i + q2 j + q3 k + q4
        Returns:
        rot_mat : np.array
            Rotation matrix (3, 3) as a xyz sequence rotation
        """
        if q.ndim == 1:
            q = q.reshape(4, 1)

        _, n = np.size(q)
        euler_angle = []

        for i in range(n):
            rot_mat = Kinematics.quatrmx(q[:, i])
            angle = Kinematics.rmxexyz(rot_mat)
            euler_angle.append(angle)
        euler_angle = np.array(euler_angle).T

        return euler_angle

    def quatezxz(q):
        """
        Parameters:
        quaternion : np.array
            Attitude quaternion [q1, q2, q3, q4] where Q = q1 i + q2 j + q3 k + q4
        Returns:
        rot_mat : np.array
            Rotation matrix (3, 3) as a zyz sequence rotation
        """
        if q.ndim == 1:
            q = q.reshape(4,1)

        _, n = np.shape(q)
        euler_angle = []

        for i in range(n):
            rot_mat = Kinematics.quatrmx(q[:, i])
            angle = Kinematics.rmxezxz(rot_mat)
            euler_angle.append(angle)
        euler_angle = np.array(euler_angle).T

        return euler_angle

    def exyzrmx(euler_angles):
        rot_mat = Kinematics.rotmaz(euler_angles[2]) @ Kinematics.rotmay(euler_angles[1]) @ Kinematics.rotmax(euler_angles[0])
        return rot_mat

    def ezxyrmx(euler_angles):
        rot_mat = Kinematics.rotmay(euler_angles[2]) @ Kinematics.rotmax(euler_angles[1]) @ Kinematics.rotmaz(euler_angles[0])
        return rot_mat

    def ezxzrmx(euler_angles):
        rot_mat = Kinematics.rotmaz(euler_angles[2]) @ Kinematics.rotmax(euler_angles[1]) @ Kinematics.rotmaz(euler_angles[0])
        return rot_mat

    def ezyxrmx(euler_angles):
        rot_mat = Kinematics.rotmax(euler_angles[2]) @ Kinematics.rotmay(euler_angles[1]) @ Kinematics.rotmaz(euler_angles[0])
        return rot_mat

    def ezxzquat(euler_angles):
        rot_mat = Kinematics.ezxzrmx(euler_angles)
        quat = Kinematics.rmxquat(rot_mat)
        return quat

    def exyzquat(euler_angles):
        rot_mat = Kinematics.exyzrmx(euler_angles)
        quat = Kinematics.rmxquat(rot_mat)
        return quat
    
    @staticmethod
    def sangvel(w):
        """
        Compute the skew-symmetric angular velocity matrix for quaternion kinematics.

        Parameters:
        w : np.array
            Angular velocity vector [wx, wy, wz]

        Returns:
        skew_ang_vel : np.array
            4x4 skew-symmetric matrix for quaternion kinematics representation.
        """
        skew_ang_vel = np.array([
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0]
        ])
        return skew_ang_vel
    
    @staticmethod
    def proximus(angleinp, angleprox):
        test = 2 * np.pi
        angle = angleprox + np.mod((angleinp - angleprox + test / 2), test) - test / 2
        return angle
    
    @staticmethod
    def rectangular_to_spherical(geoc):
        px = geoc[0]
        py = geoc[1]
        pz = geoc[2]
        ws = px * px + py * py
        rw = np.sqrt(ws + pz * pz)
        lg = np.arctan2(py, px)
        lt = np.arctan2(pz, np.sqrt(ws))
        spherical = np.array([lg, lt, rw])
        return spherical