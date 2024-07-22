import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv, eig
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# Model Parameters

l_bar = 2.0
M = 2
m = 0.3
g = 9.8

nx = 4
nu = 1

Q = np.diag([0.0, 1.0, 1.0, 0.0])
R = np.diag([0.01])

delta_t = 0.1
sim_time = 5.0
show_animation = True
save_animation = True
plot_results = True

def main():
    time = 0.0
    init_x = np.array([[0.0],
        [0.0],
        [0.3],
        [0.0]])
    A, B = get_model()
    x = init_x
    pos_traj = [init_x[0]]
    theta_traj = [init_x[2]]
    u_traj = []

    while time <= sim_time:
        
        u = lqr(x,Q,R)
        u_traj.append(u[0])
        x = sim_model(x,u)
        pos_traj.append(x[0])
        theta_traj.append(x[2])

        time += delta_t

        if show_animation:
            plt.clf()
            px = float(x[0, 0])
            theta = float(x[2, 0])
            plot_cart(px, theta)
            plt.xlim([-5.0, 2.0])
            plt.ylim([0, 3.0])  
            plt.pause(0.001)
            plt.grid()


    print("Finished")
    print(f"final x - {x[0]}, final theta = {x[2]}")
    
    if show_animation:
        plt.show()

    if save_animation:
        fig, ax = plt.subplots()
        ax.set_xlim([-5.0, 2.0])
        ax.set_ylim([0, 3.0])

        def update(frame_number):
            px = float(pos_traj[frame_number])
            theta = float(theta_traj[frame_number])
            ax.clear()
            plot_cart(px, theta)
            ax.set_xlim([-5.0, 2.0])
            ax.set_ylim([0, 3.0])  
            plt.grid()  

        ani = animation.FuncAnimation(fig, update, frames=int(sim_time/delta_t), repeat=False)
        ani.save('animation.gif', writer=PillowWriter(fps=10))
        plt.show()

    if plot_results:
        plot_pos_theta(u_traj,pos_traj,theta_traj)



def plot_pos_theta(u_traj,pos_traj,theta_traj):
    plt.figure()
    plt.plot(u_traj)
    plt.ylabel("control Input")
    plt.grid()
    plt.xlabel("time")

    plt.figure()
    plt.plot(pos_traj)
    plt.ylabel("Position")
    plt.grid()
    plt.xlabel("time")

    plt.figure()
    plt.plot(theta_traj)
    plt.ylabel("Theta")
    plt.xlabel("time")
    plt.grid()
    plt.show()

def get_model():
    A = np.array([[0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, g * (M + m) / (l_bar * M), 0.0]])

    A = np.eye(nx) + delta_t*A

    B = np.array([[0.0],
        [1.0 / M],
        [0.0],
        [1.0 / (l_bar * M)]])
    B = delta_t*B

    return A,B

def sim_model(x,u):
    A, B = get_model()
    x = A@x+B@u
    return x

def solve_DARE(A,B,Q,R, maxiter = 100, eps = 0.01):
    P = Q
    for _ in range(maxiter):
        Pn = A.T @ P @ A - A.T @ P @ B @ inv(R+B.T @ P @ B) @ B.T @ P @ A + Q 
        if (abs(Pn-P)).max() < eps:
            break
        P = Pn

    return P    

def lqr(x,Q,R):
    A, B = get_model()
    P = solve_DARE(A,B,Q,R)
    K = inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    u = -K@x
    return u

def flatten(a):
    return np.array(a).flatten()

def plot_cart(xt, theta):
    cart_w = 1.0
    cart_h = 0.5
    radius = 0.1

    cx = np.array([-cart_w / 2.0, cart_w / 2.0, cart_w /
                   2.0, -cart_w / 2.0, -cart_w / 2.0])
    cy = np.array([0.0, 0.0, cart_h, cart_h, 0.0])
    cy += radius * 2.0

    cx = cx + xt

    bx = np.array([0.0, l_bar * math.sin(-theta)])
    bx += xt
    by = np.array([cart_h, l_bar * math.cos(-theta) + cart_h])
    by += radius * 2.0

    angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
    ox = np.array([radius * math.cos(a) for a in angles])
    oy = np.array([radius * math.sin(a) for a in angles])

    rwx = np.copy(ox) + cart_w / 4.0 + xt
    rwy = np.copy(oy) + radius
    lwx = np.copy(ox) - cart_w / 4.0 + xt
    lwy = np.copy(oy) + radius

    wx = np.copy(ox) + bx[-1]
    wy = np.copy(oy) + by[-1]

    plt.plot(flatten(cx), flatten(cy), "-b")
    plt.plot(flatten(bx), flatten(by), "-k")
    plt.plot(flatten(rwx), flatten(rwy), "-k")
    plt.plot(flatten(lwx), flatten(lwy), "-k")
    plt.plot(flatten(wx), flatten(wy), "-k")
    plt.title(f"x: {xt:.2f} , theta: {math.degrees(theta):.2f}")

    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    plt.axis("equal")



if __name__ == '__main__':
    main()

