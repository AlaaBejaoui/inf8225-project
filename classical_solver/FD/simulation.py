import numpy as np
from classical_solver.FD.FiniteDifferenceSolver import FiniteDifferenceSolver


solver = FiniteDifferenceSolver(x_start=0, x_end=1, nx=10, y_start=0, y_end=1, ny=10, filename="FD_laplace")

# Set boundary conditions
solver.set_boundary(boundary_left="Dirichlet", psi_left=0, boundary_right="Dirichlet", psi_right=0,
                    boundary_down="Dirichlet", psi_down=0, boundary_up="Dirichlet", psi_up=lambda x: np.sin(np.pi*x))

# Solve and plot results
solver.solve()
solver.plot()



