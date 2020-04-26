import torch
import os
import sys
sys.path.append(os.getcwd())
from NN_solver.NN_solver_1D.NeuralNetSolver import NeuralNetSolver

torch.manual_seed(0)

# equation 1 (a trivial diff eq just to check if the code is working)
# solver = NeuralNetSolver(numHiddenLayer=0, numUnits=10, activation="leakyReLU",
#                          numEpochs=50, batch_size=5, lr=0.03,
#                          x_start=0, x_end=2, training_steps=100,
#                          testing_steps=1000,
#                          shuffle=True)  


# solver.set_diff_equation(
#     "dpsi_dx - x")
# solver.set_F("x")
# solver.set_psi_hat("0")
# solver.set_exact_solution("0.5 * np.power(x,2)")


# equation 2 (from paper page 8 problem 1)
# solver = NeuralNetSolver(numHiddenLayer=0, numUnits=10, activation="leakyReLU",
#                          numEpochs=50, batch_size=5, lr=0.03,
#                          x_start=0, x_end=1, training_steps=100,
#                          testing_steps=1000,
#                          shuffle=True)  


# solver.set_diff_equation(
#     "dpsi_dx + ( x+ (1+3*torch.pow(x,2))/(1+x+torch.pow(x,3)) ) * psi - torch.pow(x,3) - 2*x - torch.pow(x,2)*( (1+3*torch.pow(x,2))/(1+x+torch.pow(x,3)) )")
# solver.set_F("x")
# solver.set_psi_hat("1")
# solver.set_exact_solution(
#     "((np.exp(-np.power(x,2)/2) )/(1+x+np.power(x,3))) + np.power(x,2)")

# equation 3 (from paper page 8 problem 2)
solver = NeuralNetSolver(numHiddenLayer=0, numUnits=10, activation="leakyReLU",
                         numEpochs=50, batch_size=5, lr=0.03,
                         x_start=0, x_end=2, training_steps=100,
                         testing_steps=1000,
                         shuffle=True)  


solver.set_diff_equation(
    "dpsi_dx + 0.2 * psi - torch.exp(-0.2 * x) * torch.cos(x)")
solver.set_F("x")
solver.set_psi_hat("0")
solver.set_exact_solution("np.exp(-0.2 * x) * np.sin(x)") 

solver.solve()
solver.plot()
