import os
import sys
sys.path.append(os.getcwd())
from utilities.timing import time_it
from NN_solver.NN_solver_2D_hyperbolic.neuralNetwork import Net
from matplotlib import ticker, cm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch


class NeuralNetSolver:

    def __init__(self, numHiddenLayer, numUnits, activation, numEpochs, batch_size, lr, x_start, x_end, t_start, t_end, x_training_steps, t_training_steps, testing_steps, shuffle, filename):
        self.net = Net(numHiddenLayer, numUnits, activation)

        self.numEpochs = numEpochs
        self.batch_size = batch_size
        self.lr = lr
        self.x_start = x_start
        self.x_end = x_end
        self.t_start = t_start
        self.t_end = t_end
        self.x_training_steps = x_training_steps
        self.t_training_steps = t_training_steps
        self.testing_steps = testing_steps
        self.shuffle = shuffle

        self.loss_array = np.array([])
        self.psi_trial_array = np.array([])

        self.diff_equation_ = None
        self.F_ = None
        self.exact_solution_ = None
        self.psi_hat_ = None

        self.filename = filename

    @property
    def x_training_data(self):
        return torch.linspace(self.x_start, self.x_end, steps=self.x_training_steps)

    @property
    def x_testing_data(self):
        return torch.linspace(self.x_start, self.x_end, steps=self.testing_steps)

    @property
    def t_training_data(self):
        return torch.linspace(self.t_start, self.t_end, steps=self.t_training_steps)

    @property
    def t_testing_data(self):
        return torch.linspace(self.t_start, self.t_end, steps=self.testing_steps)

    @property
    def training_grid(self):
        return torch.meshgrid(self.x_training_data, self.t_training_data)

    @property
    def testing_grid(self):
        return torch.meshgrid(self.x_testing_data, self.t_testing_data)

    @property
    def training_data(self):
        return [(x_training, t_training) for x_training, t_training in zip(self.training_grid[0].reshape(-1), self.training_grid[1].reshape(-1))]

    @property
    def testing_data(self):
        return [(x_testing, t_testing) for x_testing, t_testing in zip(self.testing_grid[0].reshape(-1), self.testing_grid[1].reshape(-1))]

    @property
    def __training_loader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, shuffle=self.shuffle)

    @property
    def __testing_loader(self):
        return DataLoader(self.testing_data, batch_size=1)

    @time_it
    def solve(self):
        criterion = nn.MSELoss()
        
        # optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, nesterov=True)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        for epoch in range(self.numEpochs):

            for batch_index, (x_batch, t_batch) in enumerate(self.__training_loader):

                loss = torch.zeros([1])

                for x, t in zip(x_batch, t_batch):

                    x = x.reshape(-1)
                    t = t.reshape(-1)
                    G_ = self.G(x, t)
                    loss += criterion(G_, torch.tensor([0.]))

                loss /= len(x_batch)
                self.loss_array = np.append(self.loss_array, loss.item())

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                print(
                    f"Epoch: {epoch} - Batch: {batch_index} - Loss:{loss.item():.3e}")

    def psi_trial(self, x, t):
        return (self.psi_hat(x, t) + self.F(x, t) * self.net(torch.tensor([x, t])))

    def dpsiTrial_dx_dt(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        psi_trial_ = self.psi_trial(x, t)
        psi_trial_.backward(create_graph=True)
        grad_x = x.grad
        grad_t = t.grad
        return grad_x, grad_t

    def G(self, x, t):
        psi_trial_ = self.psi_trial(x, t)
        dpsiTrial_dx_, dpsiTrial_dt_ = self.dpsiTrial_dx_dt(x, t)
        return self.diff_equation(x, t, dpsiTrial_dx_, dpsiTrial_dt_, psi_trial_)


    # Exact solution
    def exact_solution(self, x, t):
        if not(self.exact_solution_):
            raise Exception("Exact solution not set yet!")
        return eval(self.exact_solution_)

    # Differential equation 
    def diff_equation(self, x, t, dpsi_dx, dpsi_dt, psi):
        if not(self.diff_equation_):
            raise Exception("Differential equation not set yet!")
        return eval(self.diff_equation_)

    # Function that constraints the neural network
    def F(self, x, t):
        if not(self.F_):
            raise Exception("Function F not set yet!")
        return eval(self.F_)

    # Function that satisfies the boundary conditions
    def psi_hat(self, x, t):
        if not(self.psi_hat_):
            raise Exception("Function psi hat not set yet!")
        return eval(self.psi_hat_)

    def set_exact_solution(self, command):
        self.exact_solution_ = command

    def set_diff_equation(self, command):
        self.diff_equation_ = command

    def set_F(self, command):
        self.F_ = command

    def set_psi_hat(self, command):
        self.psi_hat_ = command

    def plot(self):

        from utilities.save_results_2D import save_results_hyperbolic
        from utilities.plot_results_2D import plot_results_hyperbolic

        x =  self.testing_grid[0]
        t =  self.testing_grid[1]
        nn_solution = torch.zeros_like(x)
        for row in range(nn_solution.shape[0]):
            for column in range(nn_solution.shape[1]):
                nn_solution[row, column] = self.psi_trial(x[row, column], t[row, column])
        exact_solution = self.exact_solution(x, t)
        error = np.abs(exact_solution - nn_solution.detach().numpy())
        training_loss = self.loss_array

        save_results_hyperbolic(self.filename, x, t,  nn_solution.detach().numpy(), exact_solution, error, training_loss)
        plot_results_hyperbolic(self.filename)

