import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R

"""
Notice that difference between DCM and Rotation matrix
DCM : value(vector) conversion
Rotation matrix : axis/object transformation
"""

class Kinematics:
    def __init__(self):
        self.rad2deg = 180 / np.pi
        self.deg2rad = np.pi / 180

    @staticmethod
    def cross_matrix(w):
        """
        Skew-symmetric matrix for cross product.
        :param w: 1x3 or 3x1 numpy array (angular velocity vector)
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
        return R.from_euler('x', angle).as_matrix()

    @staticmethod
    def rotmay_rx(angle):
        """
        Rotation matrix for y-axis rotation.
        :param angle: Angle in radians
        :return: 3x3 rotation matrix
        """
        return R.from_euler('y', angle).as_matrix()

    @staticmethod
    def rotmaz_rx(angle):
        """
        Rotation matrix for z-axis rotation.
        :param angle: Angle in radians
        :return: 3x3 rotation matrix
        """
        return R.from_euler('z', angle).as_matrix()
    
    @staticmethod
    def eulerrmx(euler_angle, euler_vector):
        """
        Rodrigues' rotation formula.
        :param euler_angle: Rotation angle in radians
        :param euler_vector: 1x3 or 3x1 numpy array (rotation axis)
        :return: 3x3 rotation matrix
        """
        axis = np.asarray(euler_vector)
        axis = axis / np.linalg.norm(axis)  # 정규화 필수
        rotvec = axis * euler_angle
        return R.from_rotvec(rotvec).as_matrix()
    
    @staticmethod
    def rmxeuler(rot_mat):
        """
        Extract Rodrigues rotation parameters from a 3x3 rotation matrix using scipy.
        
        :param rot_mat: 3x3 rotation matrix
        :return: (euler_angle, euler_vector) where
                - euler_angle is the scalar rotation angle in radians
                - euler_vector is a 3D unit vector representing the rotation axis
        """
        rot = R.from_matrix(rot_mat)               # Create Rotation object
        rotvec = rot.as_rotvec()                   # Extract angle-axis representation (angle * axis)
        angle = np.linalg.norm(rotvec)             # Rotation angle
        if np.isclose(angle, 0.0):
            axis = np.array([1.0, 0.0, 0.0])        # Fallback for zero rotation
        else:
            axis = rotvec / angle                  # Normalize to get unit axis
        return angle, axis
    
    @staticmethod
    def rmxexyz(rot_mat):
        """
        Extract Euler angles from a 3x3 rotation matrix using scipy.

        :param rot_mat: (3x3 numpy array) rotation matrix
        :param order: (str) rotation sequence, e.g., 'xyz', 'zyx', 'zxz' etc.
        :return: (3x1 numpy array) Euler angles [roll,pitch,yaw] in radians
        """
        try:
            rot = R.from_matrix(rot_mat)
            euler_angles = rot.as_euler('xyz', degrees=False)
            return euler_angles
        except ValueError as e:
            print(f"Rotation extraction failed: {e}")
            return np.array([np.nan, np.nan, np.nan])
        
    @staticmethod
    def rmxezxy(rot_mat):
        """
        Extract Euler angles from a 3x3 rotation matrix using scipy.

        :param rot_mat: (3x3 numpy array) rotation matrix
        :param order: (str) rotation sequence, e.g., 'xyz', 'zyx', 'zxz' etc.
        :return: (3x1 numpy array) Euler angles [roll,pitch,yaw] in radians
        """
        try:
            rot = R.from_matrix(rot_mat)
            euler_angles = rot.as_euler('zxy', degrees=False)
            return euler_angles
        except ValueError as e:
            print(f"Rotation extraction failed: {e}")
            return np.array([np.nan, np.nan, np.nan])
    
    @staticmethod
    def rmxezxz(rot_mat):
        """
        Extract Euler angles from a 3x3 rotation matrix using scipy.

        :param rot_mat: (3x3 numpy array) rotation matrix
        :param order: (str) rotation sequence, e.g., 'xyz', 'zyx', 'zxz' etc.
        :return: (3x1 numpy array) Euler angles [roll,pitch,yaw] in radians
        """
        try:
            rot = R.from_matrix(rot_mat)
            euler_angles = rot.as_euler('zxz', degrees=False)
            return euler_angles
        except ValueError as e:
            print(f"Rotation extraction failed: {e}")
            return np.array([np.nan, np.nan, np.nan])
    
    @staticmethod
    def rmxezyx(rot_mat):
        """
        Extract Euler angles from a 3x3 rotation matrix using scipy.

        :param rot_mat: (3x3 numpy array) rotation matrix
        :param order: (str) rotation sequence, e.g., 'xyz', 'zyx', 'zxz' etc.
        :return: (3x1 numpy array) Euler angles [roll,pitch,yaw] in radians
        """
        try:
            rot = R.from_matrix(rot_mat)
            euler_angles = rot.as_euler('zyx', degrees=False)
            return euler_angles
        except ValueError as e:
            print(f"Rotation extraction failed: {e}")
            return np.array([np.nan, np.nan, np.nan])

    @staticmethod
    def rmxquat(rot_mat):
        """
        Convert rotation matrix to quaternion using scipy.
        
        Parameters:
        rot_mat : np.array
            Rotation matrix (3x3).
        
        Returns:
        quaternion : np.array (4,)
            Quaternion in (w, x, y, z) format (real part first).
        """
        rot = R.from_matrix(rot_mat)
        quat_xyzw = rot.as_quat()  # Returns in (x, y, z, w)
        quat_wxyz = np.roll(quat_xyzw, 1)  # Reorder to (w, x, y, z)
        return quat_wxyz
    
    @staticmethod
    def quat_matrix(q):
        """
        Construct the 4x4 left-side quaternion multiplication matrix Q(q)
        for q ⊗ r (Hamilton product).
        
        Parameters:
        q : (4,) array_like
            Quaternion in (w, x, y, z) format (real part first)
        
        Returns:
        Q : (4, 4) numpy.ndarray
            Left quaternion multiplication matrix
        """
        q = np.asarray(q).flatten()
        w, x, y, z = q
        return np.array([
            [w, -x, -y, -z],
            [x,  w, -z,  y],
            [y,  z,  w, -x],
            [z, -y,  x,  w]
        ])
    
    @staticmethod
    def quat_inv(q):
        """
        Compute quaternion inverse using scipy, assuming input format is [w, x, y, z].

        Parameters:
        q_wxyz : array-like of shape (4,)
            Quaternion in [w, x, y, z] format.

        Returns:
        q_inv_wxyz : ndarray of shape (4,)
            Inverted quaternion in [w, x, y, z] format.
        """
        q_wxyz = np.asarray(q).flatten()
        # Convert to scipy format [x, y, z, w]
        q_xyzw = np.roll(q_wxyz, -1)
        # Invert and convert back
        q_inv_xyzw = R.from_quat(q_xyzw).inv().as_quat()
        q_inv_wxyz = np.roll(q_inv_xyzw, 1)
        return q_inv_wxyz
    
    @staticmethod
    def quat_norm(q):
        """
        Normalize the quaternion to have a unit norm.
        q : np.array
            Input quaternion [q_real, qx, qy, qz]^T (4x1)
        Returns:
        np.array
            Normalized quaternion [q_real, qx, qy, qz]^T(4x1)
        """
        q = np.asarray(q).flatten()
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            raise ValueError("The quaternion has near-zero norm and cannot be normalized.")
        return q / norm
    
    @staticmethod
    def quat_prod(quat1, quat2):
        """
        Compute the Hamilton product (quaternion multiplication) using scipy.

        Parameters:
        quat1 : array_like (4,)
            First quaternion [w, x, y, z] format (real part first).
        quat2 : array_like (4,)
            Second quaternion [w, x, y, z] format (real part first).

        Returns:
        ndarray (4,)
            Product quaternion [w, x, y, z] format.
        """
        quat1 = np.asarray(quat1).flatten()
        quat2 = np.asarray(quat2).flatten()

        # Convert to scipy format [x, y, z, w]
        r1 = R.from_quat(np.roll(quat1, -1))
        r2 = R.from_quat(np.roll(quat2, -1))

        # Hamilton product (composition of rotations)
        r_prod = r1 * r2

        # Convert back to [w, x, y, z]
        return np.roll(r_prod.as_quat(), 1)
    
    @staticmethod
    def quatrmx(q):
        """
        Convert quaternion to rotation matrix (DCM) using scipy.

        Parameters:
        quaternion : np.array (4,)
            Quaternion in [w, x, y, z] format (real part first)

        Returns:
        rot_mat : np.array (3, 3)
            Rotation matrix (DCM)
        """
        quaternion = np.asarray(q).flatten()
        quat_xyzw = np.roll(quaternion, -1)  # Convert to [x, y, z, w]
        rot_mat = R.from_quat(quat_xyzw).as_matrix()
        return rot_mat
    
    @staticmethod
    def quatexyz(q):
        """
        Parameters:
        quaternion : np.array (4x1)
            Attitude quaternion [q1, q2, q3, q4] where Q = q1 i + q2 j + q3 k + q4
        Returns:
            euler_angle : np.array (3x1)
            Euler angles (rad) in [x, y, z] order
        """
        if q.ndim != 1:
            q = q.flatten()
            q = q.reshape(4, 1)
        
        rot_mat = Kinematics.quatrmx(q)
        angle = Kinematics.rmxexyz(rot_mat)
    
        return angle
    
    @staticmethod
    def quatezyx(q):
        """
        Parameters:
        quaternion : np.array (4x1)
            Attitude quaternion [q1, q2, q3, q4] where Q = q1 i + q2 j + q3 k + q4
        Returns:
            euler_angle : np.array (3x1)
            Euler angles (rad) in [z, y, x] order
        """
        if q.ndim != 1:
            q = q.flatten()
            q = q.reshape(4, 1)
        
        rot_mat = Kinematics.quatrmx(q)
        angle = Kinematics.rmxezyx(rot_mat)
    
        return angle
    
    @staticmethod
    def quatezxz(q):
        """
        Parameters:
        quaternion : np.array (4x1)
            Attitude quaternion [q1, q2, q3, q4] where Q = q1 i + q2 j + q3 k + q4
        Returns:
        rot_mat : np.array (3x1)
            Rotation matrix (3, 1) as a zyz sequence rotation
        """
        if q.ndim != 1:
            q = q.flatten()
            q = q.reshape(4, 1)

        rot_mat = Kinematics.quatrmx(q)
        angle = Kinematics.rmxezxz(rot_mat)

        return angle
    
    @staticmethod
    def exyzrmx(euler_angles):
        """
        Convert XYZ Euler angles to a rotation matrix (DCM) using scipy.

        Parameters:
        euler_angles : np.array (3,)
            Euler angles [roll(X), pitch(Y), yaw(Z)] in radians

        Returns:
        rot_mat : np.array (3, 3)
            Rotation matrix (DCM)
        """
        euler_angles = np.asarray(euler_angles).flatten()
        rot_mat = R.from_euler('xyz', euler_angles, degrees=False).as_matrix()
        return rot_mat
    
    @staticmethod
    def ezxyrmx(euler_angles):
        """
        Convert ZXY Euler angles to a rotation matrix (DCM) using scipy.

        Parameters:
        euler_angles : np.array (3,)
            Euler angles [roll(X), pitch(Y), yaw(Z)] in radians

        Returns:
        rot_mat : np.array (3, 3)
            Rotation matrix (DCM)
        """
        euler_angles = np.asarray(euler_angles).flatten()
        rot_mat = R.from_euler('zxy', euler_angles, degrees=False).as_matrix()
        return rot_mat
    
    @staticmethod
    def ezxzrmx(euler_angles):
        """
        Convert ZXZ Euler angles to a rotation matrix (DCM) using scipy.

        Parameters:
        euler_angles : np.array (3,)
            Euler angles [roll(X), pitch(Y), yaw(Z)] in radians

        Returns:
        rot_mat : np.array (3, 3)
            Rotation matrix (DCM)
        """
        euler_angles = np.asarray(euler_angles).flatten()
        rot_mat = R.from_euler('zxz', euler_angles, degrees=False).as_matrix()
        return rot_mat
    
    @staticmethod
    def ezyxrmx(euler_angles):
        """
        Convert ZYX Euler angles to a rotation matrix (DCM) using scipy.

        Parameters:
        euler_angles : np.array (3,)
            Euler angles [roll(X), pitch(Y), yaw(Z)] in radians

        Returns:
        rot_mat : np.array (3, 3)
            Rotation matrix (DCM)
        """
        euler_angles = np.asarray(euler_angles).flatten()
        rot_mat = R.from_euler('zyx', euler_angles, degrees=False).as_matrix()
        return rot_mat
    
    @staticmethod
    def ezxzquat(euler_angles):
        rot_mat = Kinematics.ezxzrmx(euler_angles)
        quat = Kinematics.rmxquat(rot_mat)
        return quat
    
    @staticmethod
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
    def trace(a):
        """
        Compute the trace of a square matrix.

        Parameters:
        a : np.ndarray
            Input square matrix (n x n)

        Returns:
        float
            Trace of the matrix, i.e., the sum of its diagonal elements.
        """
        return np.trace(a)
    
    @staticmethod
    def proximus(angleinp, angleprox):
        """
        Wrap input angle to be the nearest equivalent to a reference angle.

        Parameters:
        angleinp : float
            Input angle in radians.
        angleprox : float
            Reference angle to which angleinp should be closest.

        Returns:
        float
            Wrapped angle equivalent to angleinp, but closest to angleprox.
        """
        delta = angleinp - angleprox
        return angleprox + (delta + np.pi) % (2 * np.pi) - np.pi
    
    @staticmethod
    def rectangular_to_spherical(geoc):
        """
        Convert Cartesian (rectangular) coordinates to spherical coordinates.

        Parameters:
        geoc : np.ndarray
            Geocentric position [x, y, z] in meters.

        Returns:
        np.ndarray
            Spherical coordinates [longitude (rad), latitude (rad), radius (m)].
        """

        px = geoc[0]
        py = geoc[1]
        pz = geoc[2]
        ws = px * px + py * py
        rw = np.sqrt(ws + pz * pz)
        lg = np.arctan2(py, px)
        lt = np.arctan2(pz, np.sqrt(ws))
        spherical = np.array([lg, lt, rw])
        return spherical
    
    @staticmethod
    def geocentric_to_sph_geodetic(geoc):
        """
        Convert Cartesian (rectangular) coordinates to spherical coordinates.

        Parameters:
        geoc : np.ndarray
            Geocentric position [x, y, z] in meters.

        Returns:
        np.ndarray
            Spherical coordinates [longitude (rad), latitude (rad), radius (m)].
        """
        EARTH_FLATNESS	= 0.0033528131778969144; # Flattening factor = 1./298.257
        EARTH_RADIUS	= 6378139.;				 # Earth's radius in meters
        px = geoc[0]
        py = geoc[1]
        pz = geoc[2]
        gama = 1 - EARTH_FLATNESS
        gama = gama * gama
        eps = 1 - gama
        radi = EARTH_RADIUS*EARTH_RADIUS
        ws = px*px + py*py
        zs = pz * pz
        zs1 = gama*zs
        e = 1

        det = 0.01*np.sqrt((2/3)/EARTH_RADIUS)
        de = 2*det

        while (de > det):
            alf = e/(e-eps)
            zs2 = zs1 * alf * alf
            de = 0.5 * (ws + zs2 - radi*e*e)/((ws + zs2*alf)/e)
            e = e + de

        ss = (e - eps)
        ss = eps*zs/radi/ss/ss
        ro = EARTH_RADIUS*((1. + ss)/(2. + ss) + 0.25*(2. + ss))
        rw = e*ro

        arl = np.arctan2(py, px)
        sf = pz/(rw - eps*ro)
        cf = np.sqrt(ws)/rw
        anorma = np.sqrt(sf*sf + cf*cf)
        arf = np.arcsin(sf/anorma)
        geodetic = np.array([arl, arf, rw - ro])

        return geodetic
    
    @staticmethod
    def inertial_to_terrestrial(tesig, xi):
        """
        Transforms a geocentric inertial state vector into geocentric terrestrial coordinates.
        
        Parameters:
        tesig : float
            Greenwich sidereal time in radians.
        xi : numpy.ndarray
            Geocentric inertial position and velocity vector.
            [x, y, z, vx, vy, vz] (position in meters, velocity in m/s)
        
        Returns:
        numpy.ndarray
            Corresponding geocentric terrestrial vector.
        """
        R = Kinematics.rotmaz(tesig)
        
        x_pos = R @ xi[:3]
        x_vel = R @ xi[3:]
        
        return np.hstack((x_pos, x_vel))
    
    @staticmethod
    def sph_geodetic_to_geocentric(spgd):
        """
        Transforms spherical geodetic coordinates (longitude, latitude, altitude)
        into rectangular terrestrial geocentric coordinates.
        
        Parameters:
        spgd : numpy.ndarray
            Geodetic coordinates array: [longitude (rad), latitude (rad), altitude (m)]
        
        Returns:
        numpy.ndarray
            Geocentric rectangular coordinates [x, y, z] in meters.
        """
        EARTH_FLATNESS = 0.0033528131778969144  # Flattening factor = 1./298.257
        EARTH_RADIUS = 6378139.0  # Earth's radius in meters
        
        al = spgd[0]  # East longitude (radians)
        h = spgd[2]   # Altitude (meters)
        
        sf = np.sin(spgd[1])  # Sine of geodetic latitude
        cf = np.cos(spgd[1])  # Cosine of geodetic latitude
        
        gama = (1.0 - EARTH_FLATNESS) ** 2
        s = EARTH_RADIUS / np.sqrt(1.0 - (1.0 - gama) * sf ** 2)
        
        g1 = (s + h) * cf
        x = g1 * np.cos(al)
        y = g1 * np.sin(al)
        z = (s * gama + h) * sf
        
        return np.array([x, y, z])
    
    @staticmethod
    def spherical_to_rectangular(spherical):
        """
        Convert spherical coordinates to rectangular (Cartesian) coordinates.

        Parameters:
        spherical : np.ndarray
            Spherical coordinates [longitude (rad), latitude (rad), radius (m)].

        Returns:
        np.ndarray
            Cartesian coordinates [x, y, z] in meters.
        """
        clat = np.cos(spherical[1])
        geoc = np.array([
            np.cos(spherical[0])*clat,
            np.sin(spherical[0])*clat,
            np.sin(spherical[1])
        ]) * spherical[2]

        return geoc
    
    @staticmethod
    def terrestrial_to_inertial(tesig, xt):
        """
        Transforms geocentric terrestrial coordinates into geocentric inertial coordinates.
        
        Parameters:
        tesig : float
            Greenwich sidereal time in radians.
        xt : numpy.ndarray
            Geocentric terrestrial position and velocity vector.
            [x, y, z, vx, vy, vz] (position in meters, velocity in m/s)
        
        Returns:
        numpy.ndarray
            Corresponding geocentric inertial vector.
        """
        R_inv = Kinematics.rotmaz(-tesig)  # Inverse rotation
        
        x_pos = R_inv @ xt[:3]
        x_vel = R_inv @ xt[3:]
        
        return np.hstack((x_pos, x_vel))