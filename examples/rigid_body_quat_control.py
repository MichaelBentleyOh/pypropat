import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.integrate import solve_ivp
import kinematics
import ADCS
# from scipy.spatial.transform import Rotation as R
# Conversion factors
rad2deg = 180 / np.pi
deg2rad = np.pi / 180

# Time step and duration
dt = 0.01
ftime = 200
time = np.arange(0, ftime + dt, dt)

# Inertia tensor (diagonal matrix)
tensin = np.diag([10, 20, 30])
teninv = np.linalg.inv(tensin)

# Initial conditions
angle_0 = np.array([0, 0, 0]) * deg2rad
omega_0 = np.array([1, 0, 0]) * deg2rad
q_0 = np.array([-0.940,  0.001,  0.325, -0.108])  # Initial quaternion representing no rotation

# External torque
# ext_torque = np.array([0.05, 0, 0])
ext_torque = np.array([0, 0, 0])
# Magnetic moment and field (assuming zero for this example)
mag_moment = np.zeros(3)
magnetic_field = np.zeros(3)

# Set initial values
initial_state = np.hstack((q_0, omega_0))
desired_quat = np.array([1,0,0,0])
desired_angular_rate = np.array([0,0,0])
desired_state = np.concatenate([desired_quat, desired_angular_rate], 0)
# Define the equations of motion for the rigid body dynamics
def eom(t, x):
    """
    Wrapper function to call the dynamics with additional parameters.
    """
    return ADCS.rigidbody_dynamics(t, x, ext_torque, tensin, teninv, 
                                    p_gain=100,d_gain=30,
                                    # p_gain=0,d_gain=0,
                                    desired_state=desired_state,
                                    mag_moment=mag_moment,
                                    magnetic_field = magnetic_field)

# Solve using solve_ivp
sol = solve_ivp(eom, [0, ftime], initial_state, t_eval=time, method='RK45')

# Extract results
q = sol.y[0:4, :].T
w = sol.y[4:7, :].T

# Normalize quaternion
q = q / np.linalg.norm(q, axis=1)[:, None]

# Convert quaternion to Euler angles (XYZ convention)
angle = np.zeros((len(time), 3))
for i in range(len(time)):
    q0, q1, q2, q3 = q[i, :]
    angle[i, 0] = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
    angle[i, 1] = np.arcsin(2 * (q0 * q2 - q3 * q1))
    angle[i, 2] = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
    # print(q[i,:])
    # angle[i] = kinematics.quatexyz(q[i,:].T)

'''
# # Plotting results
# plt.figure(figsize=(10, 8))

# # Quaternion trajectory
# plt.subplot(4, 1, 1)
# plt.plot(time, q[:, 0], 'k', label='$q_0$')
# plt.grid()
# plt.ylabel('$q_0$')
# plt.title('Quaternion Trajectory')

# plt.subplot(4, 1, 2)
# plt.plot(time, q[:, 1], 'r', label='$q_1$')
# plt.grid()
# plt.ylabel('$q_1$')

# plt.subplot(4, 1, 3)
# plt.plot(time, q[:, 2], 'g', label='$q_2$')
# plt.grid()
# plt.ylabel('$q_2$')

# plt.subplot(4, 1, 4)
# plt.plot(time, q[:, 3], 'b', label='$q_3$')
# plt.grid()
# plt.ylabel('$q_3$')
# plt.xlabel('Time [s]')

# plt.tight_layout()
# plt.show()

# # Attitude trajectory
# plt.figure(figsize=(10, 6))
# plt.subplot(3, 1, 1)
# plt.plot(time, angle[:, 0] * rad2deg, 'r')
# plt.grid()
# plt.ylabel('Roll [deg]')
# plt.title('Attitude Trajectory')

# plt.subplot(3, 1, 2)
# plt.plot(time, angle[:, 1] * rad2deg, 'g')
# plt.grid()
# plt.ylabel('Pitch [deg]')

# plt.subplot(3, 1, 3)
# plt.plot(time, angle[:, 2] * rad2deg, 'b')
# plt.grid()
# plt.ylabel('Yaw [deg]')
# plt.xlabel('Time [s]')

# plt.tight_layout()
# plt.show()

# # Angular rate trajectory
# plt.figure(figsize=(10, 6))
# plt.subplot(3, 1, 1)
# plt.plot(time, w[:, 0] * rad2deg, 'r')
# plt.grid()
# plt.ylabel('$\omega_x$ [deg/s]')
# plt.title('Angular Rate Trajectory')

# plt.subplot(3, 1, 2)
# plt.plot(time, w[:, 1] * rad2deg, 'g')
# plt.grid()
# plt.ylabel('$\omega_y$ [deg/s]')

# plt.subplot(3, 1, 3)
# plt.plot(time, w[:, 2] * rad2deg, 'b')
# plt.grid()
# plt.ylabel('$\omega_z$ [deg/s]')
# plt.xlabel('Time [s]')

# plt.tight_layout()
# plt.show()
'''

# Define cube vertices and faces
cube_vertices = np.array([
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, -1],
    [1, -1, 1],
    [-1, 1, 1],
    [-1, 1, -1],
    [-1, -1, -1],
    [-1, -1, 1]
])
# Initialize figure
fig = plt.figure("Quaternion and Angular Velocity Animation")
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.view_init(elev=20, azim=27)
cube_faces = [
    [cube_vertices[0], cube_vertices[1], cube_vertices[2], cube_vertices[3]],
    [cube_vertices[4], cube_vertices[5], cube_vertices[6], cube_vertices[7]],
    [cube_vertices[0], cube_vertices[1], cube_vertices[5], cube_vertices[4]],
    [cube_vertices[2], cube_vertices[3], cube_vertices[7], cube_vertices[6]],
    [cube_vertices[1], cube_vertices[2], cube_vertices[6], cube_vertices[5]],
    [cube_vertices[4], cube_vertices[7], cube_vertices[3], cube_vertices[0]]
]

# Create Poly3DCollection for cube
cube_poly = Poly3DCollection(cube_faces, alpha=0.5, linewidths=1, edgecolors='k')
qlw = 2
ax.add_collection3d(cube_poly)
# ax.quiver(0,0,0,1,0,0, color='r', linewidth=qlw)
# ax.quiver(0,0,0,0,1,0, color='g', linewidth=qlw)
# ax.quiver(0,0,0,0,0,1, color='b', linewidth=qlw)
# Update function for animation
def update(i):
    # Normalize the quaternion to avoid errors due to floating point precision
    normalized_quat = q[i] / np.linalg.norm(q[i])
    
    # Convert quaternion to rotation matrix
    # rotation_matrix = R.from_quat(normalized_quat).as_matrix()
    rotation_matrix =kinematics.quatrmx(normalized_quat)
    # Apply rotation to cube vertices
    rotated_vertices = np.dot(rotation_matrix, cube_vertices.T).T
    
    # Update cube faces
    new_faces = [
        [rotated_vertices[0], rotated_vertices[1], rotated_vertices[2], rotated_vertices[3]],
        [rotated_vertices[4], rotated_vertices[5], rotated_vertices[6], rotated_vertices[7]],
        [rotated_vertices[0], rotated_vertices[1], rotated_vertices[5], rotated_vertices[4]],
        [rotated_vertices[2], rotated_vertices[3], rotated_vertices[7], rotated_vertices[6]],
        [rotated_vertices[1], rotated_vertices[2], rotated_vertices[6], rotated_vertices[5]],
        [rotated_vertices[4], rotated_vertices[7], rotated_vertices[3], rotated_vertices[0]]
    ]
    b1 = rotation_matrix @ np.array([2,0,0])
    b2 = rotation_matrix @ np.array([0,2,0])
    b3 = rotation_matrix @ np.array([0,0,2])
    # Clear old quivers
    for artist in ax.collections:
        if isinstance(artist, Line3DCollection):
            artist.remove()

    ax.quiver(0,0,0,b1[0],b1[1],b1[2], color='r', linewidth=qlw)
    ax.quiver(0,0,0,b2[0],b2[1],b2[2], color='g', linewidth=qlw)
    ax.quiver(0,0,0,b3[0],b3[1],b3[2], color='b', linewidth=qlw)
    cube_poly.set_verts(new_faces)
    return cube_poly,

# Create animation
ani = FuncAnimation(fig, update, frames=range(0, len(time), 2), interval=10, blit=False)

plt.show()