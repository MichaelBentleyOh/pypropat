import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Conversion factors
rad2deg = 180 / np.pi
deg2rad = np.pi / 180

# Time step and duration
dt = 0.01
ftime = 100
time = np.arange(0, ftime + dt, dt)

# Inertia tensor (diagonal matrix)
I_B = np.diag([10, 20, 30])

# Initial conditions
angle_0 = np.array([0, 0, 0]) * deg2rad
omega_0 = np.array([1, 0, 0]) * deg2rad
q_0 = np.array([1, 0, 0, 0])  # Initial quaternion representing no rotation

# External torque
ext_torq = np.array([0.05, 0, 0])

# Set initial values
initial_state = np.hstack((omega_0, q_0))

# Storage for results
angle = np.zeros((len(time), 3))
q = np.zeros((len(time), 4))
x = np.zeros((len(time), 3))

# Set initial values
x[0, :] = omega_0
q[0, :] = q_0
angle[0, :] = angle_0

def skew(vector):
    """
    Returns the skew-symmetric matrix of a 3-element vector.
    """
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])

def eom(t, state):
    """
    Combined equations of motion for the rigid body dynamics and kinematics.
    """
    omega = state[0:3]
    q = state[3:7]
    
    # Dynamics (angular velocity derivative)
    omega_dot = np.linalg.inv(I_B) @ (ext_torq - skew(omega) @ I_B @ omega)
    
    # Kinematics (quaternion derivative)
    q_ = q[1:]
    q0 = q[0]
    q_dot = np.zeros(4)
    q_dot[0] = -0.5 * np.dot(omega, q_)
    q_dot[1:] = -0.5 * np.cross(omega, q_) + 0.5 * q0 * omega
    
    return np.hstack((omega_dot, q_dot))

# Solve using solve_ivp
sol = solve_ivp(eom, [0, ftime], initial_state, t_eval=time, method='RK45')

# Extract results
x = sol.y[0:3, :].T
q = sol.y[3:7, :].T

# Normalize quaternion
q = q / np.linalg.norm(q, axis=1)[:, None]

# Convert quaternion to Euler angles (XYZ convention)
for i in range(len(time)):
    q0, q1, q2, q3 = q[i, :]
    angle[i, 0] = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    angle[i, 1] = np.arcsin(2 * (q0 * q2 - q3 * q1))
    angle[i, 2] = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))

# Plotting results
plt.figure(figsize=(10, 8))

# Quaternion trajectory
plt.subplot(4, 1, 1)
plt.plot(time, q[:, 0], 'k', label='$q_0$')
plt.grid()
plt.ylabel('$q_0$')
plt.title('Quaternion Trajectory')

plt.subplot(4, 1, 2)
plt.plot(time, q[:, 1], 'r', label='$q_1$')
plt.grid()
plt.ylabel('$q_1$')

plt.subplot(4, 1, 3)
plt.plot(time, q[:, 2], 'g', label='$q_2$')
plt.grid()
plt.ylabel('$q_2$')

plt.subplot(4, 1, 4)
plt.plot(time, q[:, 3], 'b', label='$q_3$')
plt.grid()
plt.ylabel('$q_3$')
plt.xlabel('Time [s]')

plt.tight_layout()
plt.show()

# Attitude trajectory
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time, angle[:, 0] * rad2deg, 'r')
plt.grid()
plt.ylabel('Roll [deg]')
plt.title('Attitude Trajectory')

plt.subplot(3, 1, 2)
plt.plot(time, angle[:, 1] * rad2deg, 'g')
plt.grid()
plt.ylabel('Pitch [deg]')

plt.subplot(3, 1, 3)
plt.plot(time, angle[:, 2] * rad2deg, 'b')
plt.grid()
plt.ylabel('Yaw [deg]')
plt.xlabel('Time [s]')

plt.tight_layout()
plt.show()

# Angular rate trajectory
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time, x[:, 0] * rad2deg, 'r')
plt.grid()
plt.ylabel('$\omega_x$ [deg/s]')
plt.title('Angular Rate Trajectory')

plt.subplot(3, 1, 2)
plt.plot(time, x[:, 1] * rad2deg, 'g')
plt.grid()
plt.ylabel('$\omega_y$ [deg/s]')

plt.subplot(3, 1, 3)
plt.plot(time, x[:, 2] * rad2deg, 'b')
plt.grid()
plt.ylabel('$\omega_z$ [deg/s]')
plt.xlabel('Time [s]')

plt.tight_layout()
plt.show()
