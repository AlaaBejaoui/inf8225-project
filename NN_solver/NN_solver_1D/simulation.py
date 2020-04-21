import torch
from NeuralNetSolver import NeuralNetSolver
from timing import time_it

torch.manual_seed(0)

solver = NeuralNetSolver(numHiddenLayer=0, numUnits=10, activation="leakyReLU", 
                        numEpochs=500, batch_size=5, lr=0.03, 
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
