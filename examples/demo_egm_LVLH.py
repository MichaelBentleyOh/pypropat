import numpy as np
import kinematics
import orbit
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp
import PIL
from matplotlib.animation import FuncAnimation

# egm_demo
# Program to demonstrate how to use the Earth Gravity Model functions
# and the numeric orbit propagator

deg2rad = np.pi / 180

### Initial condition (meters, radians) ###
kepel = np.array([7000000, 0.01, 98 * deg2rad, 0, 35 * deg2rad, 0])

stat = orbit.kepel_statvec(kepel)

delk = orbit.delkep(kepel)

year = 2017
mjd = orbit.djm(17, 7, year)

dfra = orbit.time_to_dayf(23, 0, 0)

tstart = 0
tstep = 100
tend = 6000
n = int(np.fix(tend / tstep)) + 1

orbit.egm_read_data('egm_10.dat')

# Data storage
z1 = np.zeros((1, n))
z3 = np.concatenate([z1, z1, z1], 0)
z1 = z1.squeeze(0)
r_time = copy.deepcopy(z1)
r_xo = copy.deepcopy(z3)
r_vo = copy.deepcopy(z3)
r_sma = copy.deepcopy(z1)
r_ecc = copy.deepcopy(z1)
r_inc = copy.deepcopy(z1)
r_raan = copy.deepcopy(z1)
r_par = copy.deepcopy(z1)
r_ma = copy.deepcopy(z1)
r_dist = copy.deepcopy(z1)
r_rx = copy.deepcopy(z1)
r_ry = copy.deepcopy(z1)
r_rz = copy.deepcopy(z1)

# Initialize velocity arrays
r_vx = copy.deepcopy(z1)
r_vy = copy.deepcopy(z1)
r_vz = copy.deepcopy(z1)

dist_acc = np.zeros(3)
cont_acc = np.zeros(3)

ic = 0
# Orbit propagation
for t in np.arange(tstart, tend + tstep, tstep):
    print(t)
    # Analytical orbit propagation
    kp_an = kepel + delk * t

    # Convert from Keplerian elements to state vector
    sv_an = orbit.kepel_statvec(kp_an).squeeze(0)

    xi_an = sv_an[0:3]
    vi_an = sv_an[3:6]

    # Orbit reference frame rotation matrix
    c_i_o = orbit.orbital_to_inertial_matrix(kp_an)

    tspan = np.linspace(t, t + tstep, 50)

    ext_acc = dist_acc + cont_acc

    def func(t, x, mjd=mjd, dsec=dfra, ext_acc=ext_acc):
        return orbit.egm_difeq(t, x, mjd, dsec, ext_acc)

    sol = solve_ivp(func, (t, t + tstep), tuple(stat.squeeze(0)), rtol=1e-12, atol=1e-12)
    Y = sol.y

    sv_nm = Y[:, -1]  # Propagated state vector
    xi_nm = sv_nm[0:3]  # Propagated inertial position vector
    vi_nm = sv_nm[3:6]  # Propagated inertial velocity vector
    stat = sv_nm.reshape((1, 6))  # State vector update

    # Numerically propagated Keplerian elements
    kp_nm = orbit.statvec_kepel(np.transpose(sv_nm))

    # Eccentric anomaly
    ea_nm = orbit.kepler(kp_nm[5], kp_nm[1])

    # Geocentric distance
    dist = kp_nm[0] * (1 - kp_nm[1] * np.cos(ea_nm))

    # Orbit control acceleration (if any)
    cont_acc = np.array([0, 0, 0])

    # Disturbance specific forces (if any)
    dist_acc = np.array([0, 0, 0])

    # Store data to be plotted
    r_time[ic] = t
    r_xo[:, ic] = np.dot(np.transpose(c_i_o), (xi_nm - xi_an)) / 1000
    r_vo[:, ic] = np.dot(np.transpose(c_i_o), (vi_nm - vi_an))
    r_dist[ic] = dist / 1000
    r_rx[ic] = sv_nm[0]
    r_ry[ic] = sv_nm[1]
    r_rz[ic] = sv_nm[2]
    r_sma[ic] = kp_nm[0] - kp_an[0] / 1000
    r_ecc[ic] = kp_nm[1] - kp_an[1]
    r_inc[ic] = kp_nm[2] - kp_an[2]
    r_raan[ic] = kp_nm[3] - kp_an[3]
    r_par[ic] = kp_nm[4] - kp_an[4]
    r_ma[ic] = orbit.proximus(kp_nm[5], kp_an[5]) - kp_an[5]

    # Store velocities
    r_vx[ic] = vi_nm[0]
    r_vy[ic] = vi_nm[1]
    r_vz[ic] = vi_nm[2]

    ic += 1

# Load bluemarble with PIL
bm = PIL.Image.open('earth.jpg')
# Resize and convert to array
bm = np.array(bm.resize([int(d / 2) for d in bm.size])) / 256.

# Coordinates of the image
lons = np.linspace(-180, 180, bm.shape[1]) * np.pi / 180
lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi / 180

# Create figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot Earth surface
R = 3000  # Earth radius in km
x = R * np.outer(np.cos(lons), np.cos(lats)).T
y = R * np.outer(np.sin(lons), np.cos(lats)).T
z = R * np.outer(np.ones(np.size(lons)), np.sin(lats)).T
ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=bm)

# Time setup
t = np.arange(tstart, tend + tstep, tstep)  # Include endpoint

# Plot trajectory
ax.plot3D(r_rx / 1000, r_ry / 1000, r_rz / 1000, 'k', linewidth=1)

# Create quiver objects for LVLH frame (initialized at origin)
targ_X_hlr = ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=3, label='LVLH X-axis')
targ_Y_hlr = ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=3, label='LVLH Y-axis')
targ_Z_hlr = ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=3, label='LVLH Z-axis')

# Axis settings
ax.set_xlim([-8000, 8000])
ax.set_ylim([-8000, 8000])
ax.set_zlim([-8000, 8000])
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.legend()
ax.grid(True)

# Positions (converted to km)
pos_x = r_rx / 1000
pos_y = r_ry / 1000
pos_z = r_rz / 1000

# Velocities (converted to km/s)
vel_x = r_vx / 1000
vel_y = r_vy / 1000
vel_z = r_vz / 1000

def update_quivers_LVLH(num):
    """Update the quivers in animation to represent the LVLH frame"""
    # Position and velocity vectors at current time step
    r = np.array([pos_x[num], pos_y[num], pos_z[num]])
    v = np.array([vel_x[num], vel_y[num], vel_z[num]])

    # Compute unit position vector
    r_norm = np.linalg.norm(r)
    r_hat = r / r_norm

    # Compute orbital angular momentum vector
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    h_hat = h / h_norm

    # LVLH frame axes
    z_LVLH = -r_hat
    y_LVLH = h_hat
    x_LVLH = np.cross(z_LVLH, y_LVLH)
    x_LVLH = x_LVLH / np.linalg.norm(x_LVLH)

    # Scale for visualization purposes
    axis_length = 1500  # Adjust as needed

    # Compute LVLH frame vectors
    targ_X = x_LVLH * axis_length
    targ_Y = y_LVLH * axis_length
    targ_Z = z_LVLH * axis_length

    # Update quiver segments
    targ_X_hlr.set_segments([[(r[0], r[1], r[2]),
                              (r[0] + targ_X[0], r[1] + targ_X[1], r[2] + targ_X[2])]])
    targ_Y_hlr.set_segments([[(r[0], r[1], r[2]),
                              (r[0] + targ_Y[0], r[1] + targ_Y[1], r[2] + targ_Y[2])]])
    targ_Z_hlr.set_segments([[(r[0], r[1], r[2]),
                              (r[0] + targ_Z[0], r[1] + targ_Z[1], r[2] + targ_Z[2])]])

    # Update the title with simulation time
    ax.set_title('Propagation Simulation Time : {:.2f}s'.format(t[num]))

# Creating animation
anim = FuncAnimation(fig, update_quivers_LVLH, frames=n, interval=50)

plt.show()
