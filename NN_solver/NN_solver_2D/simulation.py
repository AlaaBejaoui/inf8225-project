import torch
import os
import sys
sys.path.append(os.getcwd())
from NN_solver.NN_solver_2D.NeuralNetSolver import NeuralNetSolver

torch.manual_seed(0)

# equation 1
# solver = NeuralNetSolver(numHiddenLayer=0, numUnits=40, activation="leakyReLU",
#                          numEpochs=3, batch_size=1, lr=0.01,
#                          x_start=0, x_end=1, y_start=0, y_end=0.2, training_steps=25,
#                          testing_steps=50,
#                          shuffle=True)

# solver.set_diff_equation("x * dpsi_dx + dpsi_dy - x * y")
# solver.set_psi_hat("torch.pow(x, 2) + torch.exp(-torch.pow(x, 2))")
# solver.set_F("y")
# solver.set_exact_solution(
#     "x * (y-1) + np.power(x,2)*np.exp(-2*y) + np.exp(-np.power(x,2)*np.exp(-2*y)) + x*np.exp(-y)")

# transport equation
# solver = NeuralNetSolver(numHiddenLayer=1, numUnits=32, activation="tanh",
#                          numEpochs=1, batch_size=1, lr=0.001,
#                          x_start=-1, x_end=1, y_start=0, y_end=0.3, training_steps=100,
#                          testing_steps=50,
#                          shuffle=True)

# solver.set_diff_equation("dpsi_dy + 1 * dpsi_dx")
# solver.set_psi_hat("torch.exp(-torch.pow(x, 2)/0.1)")
# solver.set_F("y * (torch.pow(x,2) - 1)")
# solver.set_exact_solution(
#     "x * (y-1) + np.power(x,2)*np.exp(-2*y) + np.exp(-np.power(x,2)*np.exp(-2*y)) + x*np.exp(-y)")

# burger equation
solver = NeuralNetSolver(numHiddenLayer=1, numUnits=32, activation="tanh",
                         numEpochs=5, batch_size=1, lr=0.01,
                         x_start=-1, x_end=1, y_start=0, y_end=2, training_steps=100,
                         testing_steps=50,
                         shuffle=True)

solver.set_diff_equation("dpsi_dy + psi * dpsi_dx")
solver.set_psi_hat("torch.exp(-torch.pow(x, 2)/0.1)")
solver.set_F("y * (torch.pow(x,2) - 1)")
solver.set_exact_solution(
    "x * (y-1) + np.power(x,2)*np.exp(-2*y) + np.exp(-np.power(x,2)*np.exp(-2*y)) + x*np.exp(-y)")

solver.solve()
solver.plot()