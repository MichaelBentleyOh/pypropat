from core.orbit import *
from core.kinematics_new import *
import os
from core.control import *
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp
import PIL
from matplotlib.animation import FuncAnimation

# egm_demo
# Program to demonstrate how to use the Earth Gravity Model functions
# and the numeric orbit propagator

deg2rad = np.pi/180
mu = 3.986E14       # m^3/s^2

# Initial condition (meters, radians)
RE = 6731				# km
alt_init = 600			# altitude km
r_init = (RE + alt_init) * 1000		# m

incl_init = np.random.rand(1)[0] * 180 * deg2rad
w_init = np.random.rand(1)[0] * 360 * deg2rad
omega_init = np.random.rand(1)[0] * 360 * deg2rad
M_init = np.random.rand(1)[0] * 360 * deg2rad

# eccentricity 1E-5 means circular orbit
kepel_init = np.array([r_init, 1E-5, incl_init, incl_init, w_init, M_init])

stat_init = Orbit.kepel_statvec(kepel_init)

# Final Orbit
alt_final = 30000			# km
r_final = (RE + alt_final) * 1000

kepel_final = np.array([r_final, 1E-5, incl_init, incl_init, w_init, M_init])
stat_final = Orbit.kepel_statvec(kepel_final)

mean_trans = 180*deg2rad	  # rad

ratio_r = r_final / r_init
e_trans = (1 - ratio_r)/(ratio_r*np.cos(mean_trans) - 1)

a_trans = r_init / (1-e_trans)

delv1 = np.sqrt(2*mu/r_init - mu/a_trans) - np.sqrt(mu/r_init)

v_init = np.linalg.norm(stat_init[0, 3:])

stat = copy.deepcopy(stat_init)
stat[0, 3:] = stat_init[0, 3:] * (v_init + delv1) / v_init

year = 2017
mjd = Orbit.djm(17, 7, year)

dfra = Orbit.time_to_dayf(23,0,0)

tstart = 0
tstep = 100
tend1 = int(np.fix(np.pi/np.sqrt(mu/a_trans**3)/tstep))*tstep
tend2 = 100000
n = int(np.fix((tend1-tstart)/tstep))+1
n2 = int(np.fix((tend1+tend2-tstart)/tstep))+2

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, '../core/egm_10.dat')
data_path = os.path.abspath(data_path)
Orbit.egm_read_data(data_path)

# data storage
# data storage
z1      	= np.zeros((1, n))
z1_p        = np.zeros((1, n2))
z1 = z1.squeeze(0)
z1_p = z1_p.squeeze(0)
r_time      = copy.deepcopy(z1_p)
r_rx        = copy.deepcopy(z1_p)
r_ry        = copy.deepcopy(z1_p)
r_rz        = copy.deepcopy(z1_p)
r_rx_init   = copy.deepcopy(z1)
r_ry_init   = copy.deepcopy(z1)
r_rz_init   = copy.deepcopy(z1)

dist_acc = np.zeros(3)
cont_acc = np.zeros(3)

ic = 0
for t in np.arange(tstart, tstart+tend1+tstep, tstep):
    print(ic)

    ext_acc = dist_acc + cont_acc
    def func(t, x, mjd=mjd, dsec=dfra, ext_acc=ext_acc):
        return Orbit.egm_difeq(t, x, mjd, dsec, ext_acc)

    sol = solve_ivp(func, (t, t + tstep), tuple(stat.squeeze(0)), rtol=1e-12, atol=1e-12)
    Y = sol.y

    sv_nm = Y[:, -1]  # propagated state vector

    r_time[ic] = t
    r_rx[ic] = sv_nm[0]
    r_ry[ic] = sv_nm[1]
    r_rz[ic] = sv_nm[2]
    stat = sv_nm.reshape((1, 6))  # state vector update

    sol_init = solve_ivp(func, (t, t + tstep), tuple(stat_init.squeeze(0)), rtol=1e-12, atol=1e-12)
    Y = sol_init.y

    sv_nm = Y[:, -1]  # propagated state vector

    r_rx_init[ic] = sv_nm[0]
    r_ry_init[ic] = sv_nm[1]
    r_rz_init[ic] = sv_nm[2]
    stat_init = sv_nm.reshape((1, 6))  # state vector update

    ic = ic + 1

stat_final = copy.deepcopy(stat)

delv2 = np.sqrt(mu/r_final) - np.sqrt(2*mu/r_final - mu/a_trans)

v_final = np.linalg.norm(stat[0, 3:])

stat_final[0, 3:] = stat[0, 3:] * (v_final + delv2) / v_final


for t in np.arange(tend1, tend1+tend2+tstep, tstep):
    print(ic)

    ext_acc = dist_acc + cont_acc
    def func(t, x, mjd=mjd, dsec=dfra, ext_acc=ext_acc):
        return Orbit.egm_difeq(t, x, mjd, dsec, ext_acc)

    sol = solve_ivp(func, (t, t + tstep), tuple(stat_final.squeeze(0)), rtol=1e-12, atol=1e-12)
    Y = sol.y

    sv_nm = Y[:, -1]  # propagated state vector

    r_time[ic] = t
    r_rx[ic] = sv_nm[0]
    r_ry[ic] = sv_nm[1]
    r_rz[ic] = sv_nm[2]
    stat_final = sv_nm.reshape((1, 6))  # state vector update

    ic = ic + 1


# load bluemarble with PIL
bm = PIL.Image.open('earth.jpg')
# it's big, so I'll rescale it, convert to array, and divide by 256 to get RGB values that matplotlib accept
bm = np.array(bm.resize([int(d/2) for d in bm.size]))/256.

# coordinates of the image - don't know if this is entirely accurate, but probably close
lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180
lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180

# repeat code from one of the examples linked to in the question, except for specifying facecolors:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
R = 3000
x = np.outer(np.sqrt(R)*np.cos(lons), np.sqrt(R)*np.cos(lats)).T
y = np.outer(np.sqrt(R)*np.sin(lons), np.sqrt(R)*np.cos(lats)).T
z = np.outer(np.sqrt(R)*np.ones(np.size(lons)), np.sqrt(R)*np.sin(lats)).T
ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors = bm)

# Time setup
t = np.arange(tstart, tend1+tend2+tstep, tstep)  # include endpoint

# Reference vectors
ref_X = np.array([1, 0, 0])
ref_Y = np.array([0, 1, 0])
ref_Z = np.array([0, 0, 1])

# Set up the figure and axis
# ax = fig.add_subplot(111, projection='3d')
ax.plot3D(r_rx/1000, r_ry/1000, r_rz/1000, 'k', linewidth=1)  # trajectory
ax.plot3D(r_rx_init/1000, r_ry_init/1000, r_rz_init/1000, 'b', linewidth=1)  # initial orbit

# Create quiver objects
ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=1)  # X-axis
ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=1)  # Y-axis
ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=1)  # Z-axis

targ_X_hlr = ax.quiver(0, 0, 0, 1, 0, 0, color='r', linewidth=1)
targ_Y_hlr = ax.quiver(0, 0, 0, 0, 1, 0, color='g', linewidth=1)
targ_Z_hlr = ax.quiver(0, 0, 0, 0, 0, 1, color='b', linewidth=1)

# Axis settings
ax.set_xlim([-40000, 40000])
ax.set_ylim([-40000, 40000])
ax.set_zlim([-40000, 40000])
ax.grid(True)

# Positions
pos_x = r_rx/1000
pos_y = r_ry/1000
pos_z = r_rz/1000

def update_quivers(num):
    """Update the quivers in animation"""
    # theta = tend1/2*np.pi*tstep*num
    theta = 0
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    targ_X = R @ ref_X*1000
    targ_Y = R @ ref_Y*1000
    targ_Z = R @ ref_Z*1000

    targ_X_hlr.set_segments([[(pos_x[num], pos_y[num], pos_z[num]), (pos_x[num] + targ_X[0], pos_y[num] + targ_X[1], pos_z[num] + targ_X[2])]])
    targ_Y_hlr.set_segments([[(pos_x[num], pos_y[num], pos_z[num]), (pos_x[num] + targ_Y[0], pos_y[num] + targ_Y[1], pos_z[num] + targ_Y[2])]])
    targ_Z_hlr.set_segments([[(pos_x[num], pos_y[num], pos_z[num]), (pos_x[num] + targ_Z[0], pos_y[num] + targ_Z[1], pos_z[num] + targ_Z[2])]])
    ax.set_title('Propagation Simulation Time : {:.2f}s'.format(t[num]))

# Creating animation
anim = FuncAnimation(fig, update_quivers, frames=n2-1, interval=0.001)

plt.show()