import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation


# VELOCITY PROFILE FOR LANDING, Z DIRECTION
def get_desired_vel_z(z):
    if z >= 1500:
        return -30
    elif 1500 > z >= 500:
        return -20
    elif 500 > z >= 100:
        return -10
    elif 100 > z > 0:
        return -z / 10
    else:
        return 0


# Rocket and Environment Variables
rocket_mass = 10
gravity = np.array([0, 0, -9.81])
dt = 0.1
time = np.arange(0, 160+dt, dt)
frame_count = len(time)
rocket_velocity = np.ones([frame_count, 3]) * np.array([0, 0, 0])
rocket_position = np.ones([frame_count, 3]) * np.array([35, 20, 2000])
rocket_acceleration = np.zeros([frame_count, 3])
desired_vel_z = np.ones(frame_count) * 30
desired_pos_x = np.ones(frame_count) * 20
desired_pos_y = np.ones(frame_count) * 50
error_proportional = np.zeros([frame_count, 3])
error_derivative = np.zeros([frame_count, 3])
error_integral = np.zeros([frame_count, 3])
kp = np.array([12, 12, 20])
kd = np.array([6, 6, 4])
ki = np.array([1, 0, 8])


# CONTROLLER LOOP
for i in range(1, frame_count):
    desired_vel_z[i] = get_desired_vel_z(rocket_position[i-1][2])
    error_proportional[i][2] = desired_vel_z[i] - rocket_velocity[i-1][2]
    error_derivative[i][2] = (error_proportional[i][2] - error_proportional[i-1][2]) / dt
    error_integral[i][2] = error_integral[i-1][2] + error_proportional[i][2] * dt

    error_proportional[i][1] = desired_pos_y[i] - rocket_position[i-1][1]
    error_derivative[i][1] = (error_proportional[i][1] - error_proportional[i-1][1]) / dt
    error_integral[i][1] = error_integral[i-1][2] + error_proportional[i][2] * dt

    error_proportional[i][0] = desired_pos_x[0] - rocket_position[i-1][0]
    error_derivative[i][0] = (error_proportional[i][0] - error_proportional[i-1][0]) / dt
    error_integral[i][0] = error_integral[i-1][0] + error_proportional[i][0] * dt

    controller_force = error_proportional[i] * kp + error_derivative[i] * kd + error_integral[i] * ki

    if rocket_position[i-1][2] <= 0:
        gravity_force = 0
    else:
        gravity_force = gravity * rocket_mass

    force = gravity_force + controller_force
    rocket_acceleration[i] = force / rocket_mass
    rocket_velocity[i] = rocket_velocity[i-1] + 0.5 * dt * (rocket_acceleration[i] + rocket_acceleration[i-1])
    rocket_position[i] = rocket_position[i-1] + 0.5 * dt * (rocket_velocity[i] + rocket_velocity[i-1])

# create plots
np.set_printoptions(suppress=True)
fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=[0.4, 0.4, 0.4])
gs = gridspec.GridSpec(3, 4)

# Figure 0 x vs y
ax0 = fig.add_subplot(gs[0:2, 0:2])
plt.xlabel('x-position')
plt.ylabel('y-position')
ax0.xaxis.set_label_coords(0.5, 0.05)
ax0.yaxis.set_label_coords(0.05, 0.5)
plt.xlim(-20, 60)
plt.ylim(10, 90)
plt.grid()
target_position_xy = ax0.scatter(20, 50, marker='o', s=100, label='Target')
rocket_position_plot = ax0.scatter(0, 0, c='r', s=300, label='Rocket Position', marker='x')
plt.legend()


# Figure 1 x vs t
ax1 = fig.add_subplot(gs[2, 0])
plt.xlabel('time')
plt.ylabel('x-position')
plt.xlim(0, time[-1])
plt.xticks(np.arange(0, 161, 40))
plt.ylim(0, 40)
ax1.xaxis.set_label_coords(1.1, -0.2)
ax1.yaxis.set_label_coords(0.1, 0.2)
plt.grid()
x_vs_t = ax1.plot([], [], 'k', linewidth=1, label='x-position')[0]
x_vs_t_target = ax1.plot(time, desired_pos_x, 'r', linewidth=3, label='desired x-position')
plt.legend()


# Figure 2 y vs t
ax2 = fig.add_subplot(gs[2, 1])
plt.ylabel('y-position')
plt.xlim(0, time[-1])
plt.xticks(np.arange(0, 161, 40))
ax2.yaxis.set_label_coords(0.1, 0.2)
plt.ylim(0, 100)
plt.grid()
y_vs_t = ax2.plot([], [], 'k', linewidth=1, label='y-position')[0]
y_vs_t_target = ax2.plot(time, desired_pos_y, 'r', linewidth=3, label='desired y-position')
plt.legend()

# Figure 3 z vs t
ax3 = fig.add_subplot(gs[2, 2:])
plt.ylabel('z-position')
plt.xlim(0, time[-1])
plt.xticks(np.arange(0, 161, 20))
plt.ylim(0, 2200)
ax3.yaxis.set_label_coords(0.05, 0.2)
plt.grid()
z_vs_t = ax3.plot([], [], 'k', linewidth=1, label='z-position')[0]
pos_tracker_box = dict(boxstyle='square', fc=(0.1, 0.9, 0.9), ec='g', lw=1)
pos_tracker = ax3.text(135, 1650, '', color='r', size=10, bbox=pos_tracker_box)
plt.legend()
plt.xlabel('time')


# Figure 4 vel_z vs t
ax4 = fig.add_subplot(gs[0:2, 2:4])
plt.ylabel('z-velocity')
plt.ylim(10, -40)
plt.xlim(0, time[-1])
vel_tracker_box = dict(boxstyle='square', fc=(0.1, 0.9, 0.9), ec='g', lw=1)
vel_tracker = ax4.text(135, -30, '', color='r', size=10, bbox=vel_tracker_box)
vz_vs_t = ax4.plot([], [], 'k', linewidth=3, label='z-velocity')[0]
ax4.plot(time, desired_vel_z, label='desired z-velocity')
plt.legend()
plt.grid()


def update_plot(t):
    x_vs_t.set_data(time[0:t], rocket_position[:, 0][0:t])
    y_vs_t.set_data(time[0:t], rocket_position[:, 1][0:t])
    z_vs_t.set_data(time[0:t], rocket_position[:, 2][0:t])
    vz_vs_t.set_data(time[0:t], rocket_velocity[:, 2][0:t])
    rocket_position_plot.set_offsets([rocket_position[t][0], rocket_position[t][1]])
    pos_tracker.set_text(f'{round(rocket_position[:, 2][t], 2)} m')
    vel_tracker.set_text(f'{round(rocket_velocity[:,2][t], 2)} m/s')
    return x_vs_t, y_vs_t, z_vs_t, vz_vs_t, rocket_position_plot, pos_tracker, vel_tracker


rocket_animation = animation.FuncAnimation(fig, update_plot, frames=frame_count, interval=2, repeat=True, blit=True)
plt.show()
