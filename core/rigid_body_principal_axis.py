import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =======================================================================
# Author:       Yi-Hsuan Chen (Converted to Python and modified)
# Date:         06-July-2024 (Python Conversion)
# Description:  This code demonstrates the principal axes theorem by 
#               simulating the motion of a torque-free rigid body. The 
#               visualization illustrates the stability of rotations 
#               about different principal axes.
# =======================================================================

# User-defined params
AXES    = 'int'   # 'max','min','int' for principal axis selection
record  = 0        # 1: record video, 0: no record

# Numerical integration settings
tfinal  = 10
dt      = 1e-3
t_eval  = np.arange(0, tfinal+dt, dt)

# System parameters
m       = 1.0
d       = 8.0
w       = 4.0
h       = 0.5
I1      = m/12*(h**2 + w**2)
I2      = m/12*(h**2 + d**2)
I3      = m/12*(d**2 + w**2)
para    = {'I1': I1, 'I2': I2, 'I3': I3}

# Rotation matrices definition (3-2-3): psi-theta-phi
def C_C_B(phi):
    return np.array([[np.cos(phi), -np.sin(phi), 0],
                     [np.sin(phi),  np.cos(phi), 0],
                     [0,            0,           1]])

def A_C_C(th):
    return np.array([[np.cos(th), 0, np.sin(th)],
                     [0,          1,        0],
                     [-np.sin(th),0, np.cos(th)]])

def I_C_A(psi):
    return np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi),  np.cos(psi), 0],
                     [0,            0,           1]])

# Initial orientation
phi0        = 0.0
theta0      = -np.pi/2
psi0        = 0.0

# Compute initial rotation matrix
I_C_B_init  = I_C_A(psi0) @ A_C_C(theta0) @ C_C_B(phi0)
DCM         = I_C_B_init

# Principal axes vectors
qlw = 2
length = 5
e1 = np.array([length, 0, 0])
e2 = np.array([0, length, 0])
e3 = np.array([0, 0, length])
e10 = DCM @ e1
e20 = DCM @ e2
e30 = DCM @ e3

# Set initial angular velocities based on chosen axis
if AXES == 'max':
    w0 = np.array([0.001, 0, 10])
elif AXES == 'int':
    w0 = np.array([0.1, 10, 0.1])
elif AXES == 'min':
    w0 = np.array([10, 0.001, 0.001])
X0 = np.concatenate((w0, [psi0, theta0, phi0]))

# Define box vertices and faces
x = d/2
y = w/2
z = h/2
vv = np.array([[ x, -y, -z],
               [ x,  y, -z],
               [-x,  y, -z],
               [-x, -y, -z],
               [ x, -y,  z],
               [ x,  y,  z],
               [-x,  y,  z],
               [-x, -y,  z]])
fac = np.array([[1,2,6,5],
                [2,3,7,6],
                [3,4,8,7],
                [4,1,5,8],
                [1,2,3,4],
                [5,6,7,8]]) - 1  # zero-based indexing

# ODE for torque-free motion
def torqueFreeMotion(t, X, para):
    I1 = para['I1']
    I2 = para['I2']
    I3 = para['I3']
    
    w1, w2, w3, psi, theta, phi = X
    
    dX = np.zeros_like(X)
    dX[0] = w2*w3*(I2 - I3)/I1
    dX[1] = w3*w1*(I3 - I1)/I2
    dX[2] = w1*w2*(I1 - I2)/I3
    
    # Euler angle rates (3-2-3)
    dX[3] = (-w1*np.cos(phi) + w2*np.sin(phi))/np.sin(theta)
    dX[4] = w1*np.sin(phi) + w2*np.cos(phi)
    dX[5] = (w1*np.cos(phi)-w2*np.sin(phi))*1/np.tan(theta) + w3
    
    return dX

# Integrate using solve_ivp
sol = solve_ivp(torqueFreeMotion, [0, tfinal], X0, t_eval=t_eval, args=(para,), vectorized=False, rtol=1e-9, atol=1e-9)

psi   = sol.y[3,:]
theta = sol.y[4,:]
phi   = sol.y[5,:]
time  = sol.t

# Prepare figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('$x$  (m)', fontsize=12)
ax.set_ylabel('$y$  (m)', fontsize=12)
ax.set_zlabel('$z$  (m)', fontsize=12)
ax.set_xlim([-6,6])
ax.set_ylim([-6,6])
ax.set_zlim([-6,6])
ax.view_init(elev=30, azim=50)
ax.grid(True)

# Animation function
def update(frame):
    i = frame
    # Compute rotation matrix at time i
    cCB = C_C_B(phi[i])
    aCc = A_C_C(theta[i])
    ICa = I_C_A(psi[i])
    DCM = ICa @ aCc @ cCB

    b1 = DCM @ e1
    b2 = DCM @ e2
    b3 = DCM @ e3

    VV = (DCM @ vv.T).T
    poly = [VV[f] for f in fac]

    # Clear the axis and redraw everything
    ax.cla()
    # Reset the view and limits
    ax.set_xlabel('$x$ (m)', fontsize=12)
    ax.set_ylabel('$y$ (m)', fontsize=12)
    ax.set_zlabel('$z$ (m)', fontsize=12)
    ax.set_xlim([-6,6])
    ax.set_ylim([-6,6])
    ax.set_zlim([-6,6])
    ax.grid(True)

    # Add updated quivers for principal axes
    ax.quiver(0,0,0,b1[0],b1[1],b1[2], color='r', linewidth=qlw)
    ax.quiver(0,0,0,b2[0],b2[1],b2[2], color='g', linewidth=qlw)
    ax.quiver(0,0,0,b3[0],b3[1],b3[2], color='b', linewidth=qlw)

    # Add updated box
    box_new = Poly3DCollection(poly, facecolors='black', alpha=0.2)
    ax.add_collection3d(box_new)

    # Update view angle
    N = len(time)
    if i > N/4 and i < N*2/4:
        ax.view_init(elev=90, azim=0)
    else:
        ax.view_init(elev=30, azim=150)

    return []

# ani = FuncAnimation(fig, update, frames=len(time), interval=10, blit=False)
ani = FuncAnimation(fig, update, frames=range(0, len(time), 10), interval=1, blit=False)

# If you want to record the video (requires ffmpeg installed), uncomment:
if record:
    ani.save(f"{AXES}_axis_rotation.mp4", writer='ffmpeg', fps=30)

plt.show()
