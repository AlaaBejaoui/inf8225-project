import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
from NN_solver.NN_solver_1D.timing import time_it


class BurgerSolver:

    def __init__(self, x_start, x_end, N, t_start, t_end, CFL):

        self.x_start = x_start
        self.x_end = x_end
        self.t_start = t_start
        self.t_end = t_end
        self.CFL = CFL
        self.N = N
        self.u = np.zeros((self.N, 1))
        self.u_exact = np.zeros_like(self.u)
        self.u_exact_list = []  # TODO: np.array
        self.u_list = []
        self.t_list = []
        self.initial_condition = None
        self.u_left = None
        self.u_right = None
        self.x_0 = None
        self.fig, self.ax = plt.subplots()
        self.numerical_solution = None
        self.exact_solution = None
        self.animation = None

    # Compute step size
    @property
    def dx(self):
        return (self.x_end-self.x_start)/self.N

    # Grid
    @property
    def x(self):
        return np.linspace(self.x_start+self.dx/2, self.x_end-self.dx/2, self.N)

    # Set initial condition
    def set_initial_condition(self, initial_condition, u_left, u_right, x_0):
        self.initial_condition = initial_condition

        # Shock or rarefaction
        if initial_condition == "Shock" or initial_condition == "Rarefaction":
            self.u_left = u_left
            self.u_right = u_right
            self.x_0 = x_0
            # Numerical solution
            self.u[self.x < x_0] = self.u_left
            self.u[self.x >= x_0] = self.u_right
            self.u_list.append(copy.deepcopy(self.u))
            # Exact solution
            self.u_exact[self.x < x_0] = self.u_left
            self.u_exact[self.x >= x_0] = self.u_right
            self.u_exact_list.append(copy.deepcopy(self.u_exact))

        # Sine
        elif initial_condition == "Exp":
            self.u = np.exp(-np.power(self.x, 2)/0.1)
            # self.u_exact = np.sin(self.x)

        else:
            raise NotImplementedError("Initial condition not implemented")

    @time_it
    def solve(self):
        t = self.t_start
        self.t_list.append(t)

        while t <= self.t_end:
            # Compute the time step from the CFL condition
            a_max = np.max(np.abs(self.u))
            dt = (self.CFL * self.dx) / a_max

            if t + dt > self.t_end:
                dt = self.t_end - t
            if t >= self.t_end:
                break

            # Update t
            t += dt
            self.t_list.append(np.round(t, 2))

            u_new = np.zeros_like(self.u)
            for i in range(self.N):
                # Compute exact solution
                # if (self.initial_condition == "Shock") or (self.initial_condition == "Rarefaction"):
                #     self.u_exact[i] = self.exact_burger(self.u_left, self.u_right, self.x[i], t, self.x_0)

                # Compute numerical solution
                # Left boundary
                if i == 0:
                    u_l = self.u_left
                    u_r = self.u[1]
                # Right boundary
                elif i == self.N-1:
                    u_l = self.u[-2]
                    u_r = self.u[-1]
                # Inner points
                else:
                    u_l = self.u[i-1]
                    u_r = self.u[i+1]

                # Flux
                f_l = self.flux(u_l, self.u[i])
                f_r = self.flux(self.u[i], u_r)

                # Update
                u_new[i] = self.u[i] - (dt/self.dx)*(f_r-f_l)

            self.u = copy.deepcopy(u_new)

            self.u_list.append(copy.deepcopy(self.u))
            if (self.initial_condition == "Shock") or (self.initial_condition == "Rarefaction"):
                self.u_exact_list.append(copy.deepcopy(self.u_exact))

    @staticmethod
    def exact_burger(u_left, u_right, x, t, x_0):
        # Shock
        if u_left > u_right:
            # Compute the shock speed from the Rankine-Hugoniot condition
            s = (u_left + u_right)/2
            if (x-x_0)/t < s:
                u = u_left
            else:
                u = u_right

        # Rarefaction
        elif u_left < u_right:
            a_left = u_left
            a_right = u_right
            if (x-x_0)/t < a_left:
                u = u_left
            elif (x-x_0)/t > a_right:
                u = u_right
            else:
                u = (x-0.5)/t

        # Constant state: u_left = u_right
        else:
            u = u_left

        return u

    @staticmethod
    def flux(u_left, u_right):
        # Shock
        if u_left > u_right:
            # Compute shock speed
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

    def save_results(self):
        results_dict = {"u": self.u_list, "u_exact": self.u_exact_list, "t": self.t_list}
        with open("results/burger.pkl", "wb") as f:
            pickle.dump(results_dict, f)

    def plot_results(self):
        # TODO: mention problem pycharm animation
        import matplotlib.animation as animation

        with open("results/burger.pkl", "rb") as f:
            results_dict = pickle.load(f)

        t = results_dict["t"]

        self.animation = animation.FuncAnimation(self.fig, self.animate, init_func=self.init_animation,
                                                 frames=len(t), interval=10, repeat=False)

        plt.show()

    def init_animation(self):
        self.ax.grid()
        self.ax.set_xlabel("x")
        self.ax.set_xlim((self.x_start, self.x_end))
        self.ax.set_ylabel("u")
        if self.initial_condition == "Shock":
            self.ax.set_ylim((self.u_right, self.u_left))
        elif self.initial_condition == "Rarefaction":
            self.ax.set_ylim((self.u_left, self.u_right))
        elif self.initial_condition == "Sine":
            self.ax.set_ylim((-1, 1))
        self.ax.set_title(f"t = {self.t_start}")
        self.numerical_solution, = self.ax.plot([], [], "ro")
        self.exact_solution, = self.ax.plot([], [], "k")
        self.ax.legend(["Numerical solution", "Exact solution"], loc="upper left")

    def animate(self, i):
        with open("results/burger.pkl", "rb") as f:
            results_dict = pickle.load(f)

        u = results_dict["u"]
        u_exact = results_dict["u_exact"]
        t = results_dict["t"]

        self.numerical_solution.set_data(self.x, u[i])
        self.exact_solution.set_data(self.x, u_exact[i])
        self.ax.set_title(f"t = {t[i]}")
        plt.savefig(f"figures/burger_results_{str(t[i])}.svg")









