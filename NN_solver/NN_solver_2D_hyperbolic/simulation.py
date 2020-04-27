import torch
import os
import sys
sys.path.append(os.getcwd())
from NN_solver.NN_solver_2D_hyperbolic.NeuralNetSolver import NeuralNetSolver

torch.manual_seed(0)

# Equation 1 (see paper Page 4, Conservation law)
solver = NeuralNetSolver(numHiddenLayer=0, numUnits=40, activation="leakyReLU",
                         numEpochs=5, batch_size=1, lr=0.01,
                         x_start=0, x_end=1, t_start=0, t_end=0.2, x_training_steps=50, t_training_steps=20,
                         testing_steps=100,
                         shuffle=True,
                         filename = "2D_equation_1")

solver.set_diff_equation("x * dpsi_dx + dpsi_dt - x * t")
solver.set_psi_hat("torch.pow(x, 2) + torch.exp(-torch.pow(x, 2))")
solver.set_F("t * x")
solver.set_exact_solution(
    "x * (t-1) + np.power(x,2)*np.exp(-2*t) + np.exp(-np.power(x,2)*np.exp(-2*t)) + x*np.exp(-t)")

# Transport equation
# solver = NeuralNetSolver(numHiddenLayer=0, numUnits=40, activation="tanh",
#                          numEpochs=150, batch_size=1, lr=0.001,
#                          x_start=-1, x_end=1, t_start=0, t_end=0.3, x_training_steps=10, t_training_steps=5,
#                          testing_steps=100,
#                          shuffle=True,
#                          filename="2D_transport")

# solver.set_diff_equation("dpsi_dt + 1 * dpsi_dx")
# solver.set_psi_hat("0.1*torch.exp(-torch.pow(x, 2)/0.5)")
# # solver.set_F("t * (torch.pow(x,2) - 1)")
# solver.set_F("t * (x + 1)")
# solver.set_exact_solution(
#     "0.1*np.exp(-np.power((x-t), 2)/0.5)")

# Burger equation
# solver = NeuralNetSolver(numHiddenLayer=0, numUnits=40, activation="tanh",
#                          numEpochs=150, batch_size=1, lr=0.001,
#                          x_start=-1, x_end=1, t_start=0, t_end=5, x_training_steps=10, t_training_steps=50,
#                          testing_steps=100,
#                          shuffle=True,
#                          filename="2D_burger")

# solver.set_diff_equation("dpsi_dt + psi * dpsi_dx")
# solver.set_psi_hat("0.1*torch.exp(-torch.pow(x, 2)/0.5)")
# solver.set_F("t * (torch.pow(x,2) - 1)")
# # solver.set_F("t * (x + 1)")
# solver.set_exact_solution("0") # TODO: implement exact solution

solver.solve()
solver.plot()