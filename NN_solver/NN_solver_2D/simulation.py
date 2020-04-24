import torch
from NeuralNetSolver import NeuralNetSolver
from timing import time_it

torch.manual_seed(0)

solver = NeuralNetSolver(numHiddenLayer=10, numUnits=5, activation="relu",
                         numEpochs=50, batch_size=1, lr=10,
                         x_start=0.5, x_end=1, y_start=0, y_end=1, training_steps=16,
                         testing_steps=50,
                         shuffle=True)

solver.set_diff_equation("x * dpsi_dx + dpsi_dy - x * y")
solver.set_psi_hat("torch.pow(x, 2) + torch.exp(-torch.pow(x, 2))")
solver.set_F("y")

# x=torch.tensor([1.],requires_grad=True)
# y=torch.tensor([1.],requires_grad=True)
# xdx=torch.tensor([1.+1e-04],requires_grad=True)
# ydy=torch.tensor([1.+1e-04],requires_grad=True)

# print(solver.psi_trial(x,y))
# print(solver.psi_trial(xdx,y))
# print(solver.dpsiTrial_dx(x,y))
# print()
# print(solver.psi_trial(x,y))
# print(solver.psi_trial(x,ydy))
# print(solver.dpsiTrial_dy(x,y))

solver.solve()
solver.set_exact_solution(
    "x * (y-1) + x*x*np.exp(-2*y) + np.exp(-x*x*np.exp(-2*y)) + x*np.exp(-y)")

solver.plot()

# import torch

# x = torch.tensor([[1,2,3,4,5],
#         [6,7,8,9,10],
#         [11,12,13,14,15],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])

# x[2,3] = 56.

# print(x)
