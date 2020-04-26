import torch
import os
import sys
sys.path.append(os.getcwd())
from NN_solver.NN_solver_2D_elliptic.NeuralNetSolver import NeuralNetSolver

torch.manual_seed(0)

# Laplace equation
solver = NeuralNetSolver(numHiddenLayer=0, numUnits=16, activation="tanh",
                         numEpochs=15, batch_size=1, lr=0.005,
                         x_start=0, x_end=1, y_start=0, y_end=1, x_training_steps=5, y_training_steps=5,
                         testing_steps=100,
                         shuffle=True,
                         filename="2D_laplace")

solver.set_diff_equation("dpsi_dx2 + dpsi_dy2")
solver.set_psi_hat("y * torch.sin(3.14*x)")
solver.set_F("x * (x-1) * y * (y-1)")
solver.set_exact_solution(
    "(1/(np.exp(np.pi)-np.exp(-np.pi))) * np.sin(np.pi*x) * (np.exp(np.pi*y)-np.exp(-np.pi*y))") 

solver.solve()
solver.plot()