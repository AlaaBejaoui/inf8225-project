import numpy as np
import matplotlib.pyplot as plt
import copy
from NN_solver.NN_solver_1D.timing import time_it


class BurgerSolver:

    def __init__(self, x_start, x_end, N, t_start, t_end, CFL):

        self.x_start = x_start
        self.x_end = x_end
        self.t_start = t_start
        self.t_end = t_end
        self.CFL = CFL
        self.N = N
        self.u = np.zeros((self.N+1, 1))
        self.u_old = np.zeros_like(self.u)
        self.u_exact = np.zeros((self.N+1, 1))
        self.u_left = None
        self.u_right = None
        self.u_exact_list = None
        self.u_list = None
        self.t_list = None
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.ani = None
        self.numerical_solution = None
        self.exact_solution = None

    # Compute step size
    @property
    def dx(self):
        return (self.x_end-self.x_start)/self.N

    # Grid
    @property
    def x(self):
        # return self.x = np.linspace(self.x_start+self.dx/2, self.x_end-self.dx/2, N)
        return np.linspace(self.x_start-self.dx, self.x_end+self.dx, self.N+1)

    @property
    def x_exact(self):
        return np.linspace(self.x_start, self.x_end, self.N)

    # Set initial condition
    def set_initial_condition(self, initial_condition, u_left, u_right):
        if initial_condition == "Shock" or initial_condition == "Rarefaction":
            self.u_left = u_left
            self.u_right = u_right
            self.u_old[self.x < 0.5] = self.u_left
            self.u_exact[self.x < 0.5] = self.u_left
            self.u_old[self.x >= 0.5] = self.u_right
            self.u_exact[self.x >= 0.5] = self.u_right
            self.u_list = [self.u_old]
            self.u_exact_list = [self.u_exact]
        elif initial_condition == "Sine":
            self.u_old = np.sin(self.x)
            self.u_exact = np.sin(self.x)
        else:
            raise NotImplementedError("Initial condition not implemented")

    @time_it
    def solve(self):
        t = self.t_start
        self.t_list = [t]
        j = 0
        while t <= self.t_end:
            j += 1
            # Compute the exact solution
            for i in range(self.N):
                self.u_exact[i] = self.exact_burger(self.u_left, self.u_right, self.x_exact[i], t)

            self.u_exact_list.append(self.u_exact)
            # self.u_exact_list = np.column_stack((self.u_exact_list, self.u_exact))

            # Compute the time step from the CFL condition
            # a_max = np.max(np.abs(u))
            # dt = (CFL*dx)/a_max
            dt = 0.001

            # if t+dt > self.t_end:
            #    dt = self.t_end-t
            # if t >= self.t_end:
            #    break

            t += dt
            self.t_list.append(np.round(t, 2))

            self.u_old[0] = self.u_left
            self.u_old[-1] = self.u_old[-2]

            for i in range(1, self.N):
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
                f_l = self.gudonov_flux(self.u_old[i-1], self.u_old[i])
                f_r = self.gudonov_flux(self.u_old[i], self.u_old[i + 1])

                # Update
                self.u[i] = self.u_old[i] - (dt/self.dx)*(f_r-f_l)

            self.u_old = copy.deepcopy(self.u)
            self.u_list.append(self.u)
            # self.u_list = np.column_stack((self.u_list, self.u))

    @staticmethod
    def exact_burger(u_left, u_right, x, t):
        # Shock
        if u_left > u_right:
            # Compute the shock speed from the Rankine-Hugoniot condition
            s = (u_left + u_right)/2
            if s*t > (x-0.5):
                u = u_left
            else:
                u = u_right
        # Rarefaction
        elif u_left < u_right:
            a_left = u_left
            a_right = u_right
            if a_left*t > (x-0.5):
                u = u_left
            elif a_right*t < (x-0.5):
                u = u_right
            else:
                u = (x-0.5)/t
        # Constant state: u_left = u_right
        else:
            u = u_left
        return u

    @staticmethod
    def gudonov_flux(u_left, u_right):
        # Shock
        if u_left > u_right:
            # Compute the shock speed from the Rankine-Hugoniot condition
            s = (u_left+u_right)/2
            if s > 0:
                u = u_left
            else:
                u = u_right
        # Rarefaction
        elif u_left < u_right:
            a_left = u_left
            a_right = u_right
            if a_left > 0:
                u = u_left
            elif a_right < 0:
                u = u_right
            else:
                u = 0
        # Constant state: u_left = u_right
        else:
            u = u_left
        
        f = 0.5*u**2

        return f

    def plot_results(self):
        import matplotlib
        import matplotlib.animation as animation

        # matplotlib.use("TkAgg")

        self.ani = animation.FuncAnimation(self.fig, self.animate, init_func=self.init_animation,
                                           frames=1000, interval=10)
        plt.show()

    def init_animation(self):
        self.ax.grid()
        self.ax.set_xlabel("x")
        self.ax.set_xlim((self.x_start, self.x_end))
        self.ax.set_ylabel("u")
        self.ax.set_ylim((0, 2))
        self.ax.set_title(f"t = {self.t_start}")
        self.numerical_solution, = self.ax.plot([], [], "ro")
        self.exact_solution, = self.ax.plot([], [], "k")
        # self.ax.legend(["Numerical solution", "Exact solution"])

    def animate(self, i):
        y = self.u_list[:, i]
        y_exact = self.u_exact_list[:, i]
        self.numerical_solution.set_data(self.x, y)
        self.exact_solution.set_data(self.x_exact, y_exact)
        self.ax.set_title(f"t = {self.t_list[i]}")









