import numpy as np
import pickle
from NN_solver.NN_solver_1D.timing import time_it


class LaPlaceSolver:

    def __init__(self, x_start, x_end, nx, y_start, y_end, ny, lambda_, rho, c):
        
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.nx = nx
        self.ny = ny
        self.lambda_ = lambda_
        self.rho = rho
        self.c = c
        self.T = np.zeros((nx*ny, 1))
        self.T_down = None

    # Compute step size along the x-axis
    @property
    def dx(self):
        return (self.x_end-self.x_start)/(self.nx-1)

    # Compute step size along the y-axis
    @property
    def dy(self):
        return (self.y_end-self.y_start)/(self.ny-1)

    # Grid
    @property
    def x(self):
        return np.linspace(self.x_start, self.x_end, self.nx)

    @property
    def y(self):
        return np.linspace(self.y_start, self.y_end, self.ny)

    # Compute thermal diffusivity
    @property
    def a(self):
        return self.lambda_/(self.rho*self.c)

    # Set boundary conditions
    def set_boundary(self, boundary_left, boundary_right, boundary_down, T_down, boundary_up):
        if boundary_down == "Dirichlet":
            self.T_down = T_down
        else:
            raise NotImplementedError("Boundary condition not implemented")
        if boundary_up == "Dirichlet":
            pass
        else:
            raise NotImplementedError("Boundary condition not implemented")
        if boundary_left == "Neumann":
            pass
        else:
            raise NotImplementedError("Boundary condition not implemented")
        if boundary_right == "Neumann":
            pass
        else:
            raise NotImplementedError("Boundary condition not implemented")


    @time_it
    def solve(self):
        # Initialization
        A = np.zeros((self.nx*self.ny, self.nx*self.ny))
        r = np.zeros((self.nx*self.ny, 1))
        
        for j in range(self.ny):
            for i in range(self.nx):
                i_0 = i+j*self.nx
                i_left = i_0-1
                i_right = i_0+1
                i_up = i_0+self.nx
                i_down = i_0-self.nx
                
                # Inner points
                if (0 < i) and (i < self.nx-1) and  (0 < j) and (j < self.ny-1):
                    A[i_0, i_0] = -2*self.a/self.dx**2-2*self.a/self.dy**2
                    A[i_0, i_up] = self.a/self.dy**2
                    A[i_0, i_down] = self.a/self.dy**2
                    A[i_0, i_right] = self.a/self.dx**2
                    A[i_0, i_left] = self.a/self.dx**2
                    r[i_0] = 0

                # Boundary
                else:
                    # Upper boundary
                    if j == self.ny-1:
                        A[i_0, i_0] = 1
                        r[i_0] = self.f(self.x[i])
                    
                    # Lower boundary
                    if j == 0:
                        A[i_0, i_0] = 1
                        r[i_0] = self.T_down

                    # Left and right boundary
                    if (0 < j) and (j < self.nx-1): 
                        # Left boundary
                        if i == 0:
                            A[i_0, i_0] = 1
                            A[i_0, i_right] = -1
                        # Right boundary
                        if i == self.nx-1: 
                            A[i_0, i_0] = 1
                            A[i_0, i_left] = -1
             
        # Solve linear system of equations
        self.T = np.linalg.solve(A, r)

    @staticmethod
    def f(x):
        return 300 - 80 * np.exp(-0.5 * (x / 0.3) ** 2)

    def save_results(self):
        with open("results/laplace.pkl", "wb") as f:
            pickle.dump(self.T, f)

    def plot_results(self):
        import matplotlib.pyplot as plt
        with open("results/laplace.pkl", "rb") as f:
            T = pickle.load(f)
        T_plot = np.reshape(T, (self.nx, self.ny))
        fig, ax = plt.subplots()
        contour_plot = ax.contourf(self.x, self.y, T_plot, levels=20, cmap="coolwarm")
        ax.contour(contour_plot, linewidths=1, colors="k")
        ax.set_title("FD solution of the LaPlace equation")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        color_bar = plt.colorbar(contour_plot)
        color_bar.set_label("T[K]")
        plt.savefig('figures/laplace_results.svg')
        plt.show()