import numpy as np
from utilities.timing import time_it


class FiniteDifferenceSolver:

    def __init__(self, x_start, x_end, nx, y_start, y_end, ny, filename):

        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.nx = nx
        self.ny = ny
        self.psi = np.zeros((nx*ny, 1))
        self.psi_left = None
        self.psi_right = None
        self.psi_up = None
        self.psi_down = None
        self.filename = filename

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

    # Set boundary conditions
    def set_boundary(self, boundary_left, psi_left, boundary_right, psi_right, boundary_down, psi_down,
                     boundary_up, psi_up):
        if boundary_down == "Dirichlet":
            self.psi_down = psi_down
        else:
            raise NotImplementedError("Boundary condition not implemented")
        if boundary_up == "Dirichlet":
            self.psi_up = psi_up(self.x)
        else:
            raise NotImplementedError("Boundary condition not implemented")
        if boundary_left == "Dirichlet":
            self.psi_left = psi_left
        else:
            raise NotImplementedError("Boundary condition not implemented")
        if boundary_right == "Dirichlet":
            self.psi_right = psi_right
        else:
            raise NotImplementedError("Boundary condition not implemented")

    @time_it
    def solve(self):
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
                if (0 < i) and (i < self.nx-1) and (0 < j) and (j < self.ny-1):
                    A[i_0, i_0] = -2/np.power(self.dx, 2)-2/np.power(self.dy, 2)
                    A[i_0, i_up] = 1/np.power(self.dy, 2)
                    A[i_0, i_down] = 1/np.power(self.dy, 2)
                    A[i_0, i_right] = 1/np.power(self.dx, 2)
                    A[i_0, i_left] = 1/np.power(self.dx, 2)
                    r[i_0] = 0

                # Boundary
                else:
                    # Upper boundary
                    if j == self.ny-1:
                        A[i_0, i_0] = 1
                        r[i_0] = self.psi_up[i]
                    
                    # Lower boundary
                    if j == 0:
                        A[i_0, i_0] = 1
                        r[i_0] = self.psi_down

                    # Left boundary
                    if i == 0:
                        A[i_0, i_0] = 1
                        r[i_0] = self.psi_left

                    # Right boundary
                    if i == self.nx-1:
                        A[i_0, i_0] = 1
                        r[i_0] = self.psi_right
             
        # Solve linear system of equations
        self.psi = np.linalg.solve(A, r)

    # Compute exact solution
    @property
    def exact_solution(self):
        X, Y = np.meshgrid(self.x, self.y)
        return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * np.sin(np.pi * X) * (np.exp(np.pi * Y) - np.exp(-np.pi * Y))

    def plot(self):

        from utilities.save_results_2D import save_results_FD
        from utilities.plot_results_2D import plot_results_FD

        X = self.x
        Y = self.y
        FD_solution = np.reshape(self.psi, (self.nx, self.ny))
        error = np.abs(self.exact_solution - FD_solution)
        save_results_FD(self.filename, X, Y, FD_solution, error)
        plot_results_FD(self.filename)



