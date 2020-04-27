from classical_solver.BurgerSolver import BurgerSolver

solver = BurgerSolver(x_start=0, x_end=1, N=100, t_start=0, t_end=0.5, CFL=0.2)

# Set initial condition
solver.set_initial_condition(initial_condition="Exp", u_left=0, u_right=2, x_0=0.5)

# Solve Burger equation
print("Solving the Burger equation ...")
solver.solve()

# Save results
print("Saving the results ...")
solver.save_results()

# Plot results
solver.plot_results()
