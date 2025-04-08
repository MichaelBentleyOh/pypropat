import numpy as np
from kinematics import Kinematics as kin

class Control:
    def __init__(self):
        self.rad2deg = 180 / np.pi
        self.deg2rad = np.pi / 180
    
    @staticmethod
    def triad(v, w):
        nz = np.linalg.norm(v)

        if nz == 0:
            c_triad = np.eye(3)
        else:
            x = v / np.linalg.norm(v)
            z = np.cross(x, w)
            if np.linalg.norm(z) == 0:
                raise ValueError("Vectors v and w are collinear or one of them is zero.")
            z = z / np.linalg.norm(z)
            y = np.cross(z, x)
            c_triad = np.array([x, y, z]).T

        return c_triad
    
    @staticmethod
    def rigidbody_dynamics(t, x, ext_torque, tensin, 
                       mag_moment=None, magnetic_field=None,
                       p_gain=0, d_gain=0, desired_state=None):
        if mag_moment is None:
            mag_moment = np.zeros(3)
        if magnetic_field is None:
            magnetic_field = np.zeros(3)
        if desired_state is None:
            desired_state = np.array([1, 0, 0, 0, 0, 0, 0])
        
        quat = x[0:4]
        omega = x[4:7]

        p_err = kin.quat_prod(quat, kin.quat_inv(desired_state[0:4]))
        p_err = p_err[1:4]
        d_err = omega - desired_state[4:7] 

        control_torque = -p_gain*p_err -d_gain*d_err

        # Dynamics (angular velocity derivative)
        omega_dot = np.linalg.inv(tensin) @ (ext_torque + control_torque - kin.cross_matrix(omega) @ tensin @ omega)

        # Kinematics (quaternion derivative)
        q_vec = quat[1:]
        q0 = quat[0]
        q_dot = np.zeros(4)
        q_dot[0] = -0.5 * np.dot(omega, q_vec)
        q_dot[1:] = -0.5 * np.cross(omega, q_vec) + 0.5 * q0 * omega

        return np.hstack((q_dot,omega_dot))