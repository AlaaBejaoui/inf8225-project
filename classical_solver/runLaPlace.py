from classical_solver.LaPlaceSolver import LaPlaceSolver


solver = LaPlaceSolver(x_start=-1, x_end=1, nx=21, y_start=-1, y_end=1, ny=21, lambda_=0.0262, rho=1.293, c=1005)

# Set boundary conditions
solver.set_boundary(boundary_left="Neumann", boundary_right="Neumann", boundary_down="Dirichlet", T_down=400,
                    boundary_up="Dirichlet")
# Solve LaPlace equation
print("Solving the LaPlace equation ...")
solver.solve()

# Save results
print("Saving the results ...")
solver.save_results()

# Plot results
solver.plot_results()



