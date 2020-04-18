import numpy as np
import matplotlib.pyplot as plt
from Laplace_FD import Laplace_FD

# ...
Laplace_FD = Laplace_FD(x_start=-1, x_end=1, y_start=-1, y_end=1, nx=21, ny=21, lambda_=0.0262, rho=1.293, c=1005, 
    BC_left="Neumann", BC_right="Neumann", BC_down="Dirichlet", T_down=400, BC_up="Dirichlet", T_up=400)

# Solve Laplace equation
Laplace_FD.solve()
print("Solving the Laplace equation ...")

# Plot results
T_plot = np.reshape(Laplace_FD.T, (Laplace_FD.nx, Laplace_FD.ny))
fig, ax = plt.subplots()
contourPlot = ax.contourf(Laplace_FD.x, Laplace_FD.y, Laplace_FD.T, levels=20, cmap="coolwarm")
ax.contour(contourPlot, linewidths=1, colors="k")
ax.set_title("Title")
ax.set_xlabel("x")
ax.set_ylabel("y")
colorBar = plt.colorbar(contourPlot)
colorBar.set_label("T[K]")
plt.show()


