import numpy as np
import copy
from utilities.timing import time_it


class FiniteVolumeSolver:

    def __init__(self, equation, x_start, x_end, N, t_start, t_end, CFL, filename, c=None):

        self.equation = equation
        self.c = c
        self.x_start = x_start
        self.x_end = x_end
        self.t_start = t_start
        self.t_end = t_end
        self.CFL = CFL
        self.N = N
        self.psi = np.zeros((self.N, 1))
        self.psi_array = np.array([])
        self.t_array = np.array([])
        self.psi_left = None
        self.filename = filename

    # Compute step size
    @property
    def dx(self):
        return (self.x_end-self.x_start)/self.N

    # Grid
    @property
    def x(self):
        return np.linspace(self.x_start+self.dx/2, self.x_end-self.dx/2, self.N)

    # Set initial condition
    def set_initial_and_boundary_condition(self, initial_condition, psi_initial, boundary_left):
        if initial_condition == "Gauss":
            self.psi = psi_initial(self.x)
            self.psi_array = copy.deepcopy(self.psi)
        else:
            raise NotImplementedError("Initial condition not implemented")
        if boundary_left == "Dirichlet":
            self.psi_left = psi_initial(self.x_start)
        else:
            raise NotImplementedError("Boundary condition not implemented")

    @time_it
    def solve(self):
        t = self.t_start
        self.t_array = np.append(self.t_array, t)

        while t <= self.t_end:
            # Compute time step from the CFL condition
            if self.equation == "Transport":
                a = self.c
            elif self.equation == "Burger":
                a = np.max(np.abs(self.psi))
            else:
                raise NotImplementedError("Different equation not implemented")

            dt = (self.CFL * self.dx) / a

            if t + dt > self.t_end:
                dt = self.t_end - t
            if t >= self.t_end:
                break

            # Update t
            t += dt
            self.t_array = np.append(self.t_array, np.round(t, 2))

            psi_new = np.zeros_like(self.psi)
            for i in range(self.N):

                # Left boundary
                if i == 0:
                    psi_l = self.psi_left
                    psi_r = self.psi[1]
                # Right boundary
                elif i == self.N-1:
                    psi_l = self.psi[-2]
                    psi_r = self.psi[-1]
                # Inner points
                else:
                    psi_l = self.psi[i-1]
                    psi_r = self.psi[i+1]

                # Flux
                f_l = self.flux(psi_l, self.psi[i])
                f_r = self.flux(self.psi[i], psi_r)

                # Update
                psi_new[i] = self.psi[i] - (dt/self.dx)*(f_r-f_l)

            self.psi = copy.deepcopy(psi_new)
            self.psi_array = np.column_stack((self.psi_array, self.psi))

    def flux(self, psi_left, psi_right):
        # Shock
        if psi_left > psi_right:
            # Compute shock speed
            s = (psi_left+psi_right)/2
            if s > 0:
                psi = psi_left
            else:
                psi = psi_right

        # Rarefaction
        elif psi_left < psi_right:
            a_left = psi_left
            a_right = psi_right
            if a_left > 0:
                psi = psi_left
            elif a_right < 0:
                psi = psi_right
            else:
                psi = 0

        # Constant state: psi_left = psi_right
        else:
            psi = psi_left

        # Compute flux
        if self.equation == "Transport":
            f = self.c*psi
        elif self.equation == "Burger":
            f = 0.5*np.power(psi, 2)
        else:
            raise NotImplementedError("Different equation not implemented")

        return f

    # Compute exact solution
    @property
    def exact_solution(self):
        if self.equation == "Transport":
            x, t = np.meshgrid(self.x, self.t_array)
            return 0.1*np.exp(-np.power((x-t), 2)/0.5)
        elif self.equation == "Burger":
            return None
        else:
            raise NotImplementedError("Different equation not implemented")

    def plot(self):

        from utilities.save_results_2D import save_results_FV
        from utilities.plot_results_2D import plot_results_FV

        x = self.x
        t = self.t_array
        FV_solution = self.psi_array.transpose()
        if self.exact_solution is not None:
            error = np.abs(self.exact_solution - FV_solution)
            save_results_FV(self.filename, x, t, FV_solution, error)
        else:
            save_results_FV(self.filename, x, t, FV_solution)
        plot_results_FV(self.filename)








