import numpy as np
from classical_solver.FV.FiniteVolumeSolver import FiniteVolumeSolver

# Transport equation
# solver = FiniteVolumeSolver(equation="Transport", c=1, x_start=-1, x_end=1, N=100, t_start=0, t_end=0.3,
#                             CFL=0.9, filename="FV_transport")

# Burger equation
solver = FiniteVolumeSolver(equation="Burger", x_start=-1, x_end=1, N=100, t_start=0, t_end=5,
                            CFL=0.9, filename="FV_burger")

# Set initial and boundary condition
solver.set_initial_and_boundary_condition(initial_condition="Gauss",
                                          psi_initial=lambda x: 0.1*np.exp(-np.power(x, 2)/0.5),
                                          boundary_left="Dirichlet")

# Solve and plot results
solver.solve()
solver.plot()
