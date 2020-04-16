import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy

matplotlib.use("TkAgg")


def godFlux(u_l, u_r):
    # Shock
    if u_l > u_r:
        # Compute the shock speed from the Rankine-Hugoniot condition
        s = (u_l+u_r)/2
        if s > 0:
            u = u_l
        else:
            u = u_r
    # Rarefaction
    elif u_l < u_r:
        a_l = u_l
        a_r = u_r
        if a_l > 0:
            u = u_l
        elif a_r < 0:
            u = u_r
        else:
            u = 0
    # u_l = u_r
    else:
        u = u_l
        
    f = 0.5*u**2
    return f


def exactBurger(u_l, u_r, x, t):
    # Shock
    if u_l > u_r:
        # Compute the shock speed from the Rankine-Hugoniot condition
        s = (u_l + u_r)/2
        if s*t > (x-0.5):
            u = u_l
        else:
            u = u_r
    # Rarefaction
    elif u_l < u_r:
        a_l = u_l
        a_r = u_r
        if a_l*t > (x-0.5):
            u = u_l
        elif a_r*t < (x-0.5):
            u = u_r
        else:
            u = (x-0.5)/t
    # u_l = u_r
    else:
        u = u_l
        
    return u


# Parameters
x_0 = 0
x_1 = 1
t_0 = 0
t_1 = 0.5
u_1 = 0
u_2 = 2
CFL = 0.2
initial_condition = "rarefaction"


# Grid
Ncell = 100
Nexact= 500
dx = (x_1-x_0)/Ncell
# x = np.linspace(x_0+dx/2, x_1-dx/2, Ncell)
x = np.linspace(x_0-dx, x_1+dx, Ncell+1)
u = np.zeros((Ncell+1, 1))
u_old = np.zeros_like(u)

# Exact solution
x_exact = np.linspace(x_0, x_1, Nexact)
u_exact = np.zeros((Nexact, 1))

t = t_0
t_list = [t]

# Initial condition
if initial_condition == "shock":
    u[x<0] = u_2
    u[x>=0] = u_1
elif initial_condition == "rarefaction":
    for i in range(Ncell):
        if x[i] < 0.5:
            u_old[i] = u_1
            u_exact[i] = u_1
        else:
            u_old[i] = u_2
            u_exact[i] = u_2
elif initial_condition == "sine":
    u = np.sin(x)
else:
    raise NotImplementedError("Initial condition not implemented")

u_list = u_old
u_exact_list = u_exact

# Plot initial condition
fig = plt.figure()
ax = plt.axes()
ax.grid()
ax.set_xlabel("x")
ax.set_xlim((x_0, x_1))
ax.set_ylabel("u")
ax.set_ylim((0, 2))
ax.set_title(f"t = {t_0}")
numerical_solution, = ax.plot(x, u_old, "ro")
exact_solution, = ax.plot(x_exact, u_exact, "k")
# ax.legend(["Numerical solution", "Exact solution"])


def animate(i):
    y = u_list[:, i]
    y_exact = u_exact_list[:, i]
    numerical_solution.set_data(x, y)
    exact_solution.set_data(x_exact, y_exact)
    ax.set_title(f"t = {t_list[i]}")
    return numerical_solution, exact_solution


while t <= t_1:
    for i in range(Nexact):
        u_exact[i] = exactBurger(u_1, u_2, x_exact[i], t)
    u_exact_list = np.column_stack((u_exact_list, u_exact))
        
    # Compute the time step from the CFL condition
    # a_max = np.max(np.abs(u))
    # dt = (CFL*dx)/a_max
    dt = 0.001
    
    # if t+dt > t_1:
    #    dt = t_1-t
    # if t >= t_1:
    #    break 
    
    t += dt
    t_list.append(np.round(t, 2))
    
    u_old[0] = u_1
    u_old[-1] = u_old[-2]

    for i in range(1, Ncell):
        # Left boundary
        # if i == 0:
        #    u_l = u_1 # u[-1] 
        #    u_r = u[1]
        # Right boundary
        # elif i == Ncell-1:
        #    u_l = u[-2]
        #    u_r = u[-1] # u[0] 
        # Inner points
        # else:
        #    u_l = u[i-1]
        #    u_r = u[i+1]

        # Flux
        f_l = godFlux(u_old[i-1], u_old[i])    
        f_r = godFlux(u_old[i], u_old[i+1])

        # Update
        u[i] = u_old[i] - (dt/dx)*(f_r-f_l)

    u_old = copy.deepcopy(u)
    u_list = np.column_stack((u_list, u))


# Plot
ani = animation.FuncAnimation(fig, animate, frames=np.arange(len(t_list)), interval=10)
plt.show()

