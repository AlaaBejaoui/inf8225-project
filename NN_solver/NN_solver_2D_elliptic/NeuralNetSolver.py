import os
import sys
sys.path.append(os.getcwd())
from utilities.timing import time_it
from NN_solver.NN_solver_2D_elliptic.neuralNetwork import Net
from matplotlib import ticker, cm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch


class NeuralNetSolver:

    def __init__(self, numHiddenLayer, numUnits, activation, numEpochs, batch_size, lr, x_start, x_end, y_start, y_end, x_training_steps, y_training_steps, testing_steps, shuffle, filename):
        self.net = Net(numHiddenLayer, numUnits, activation)

        self.numEpochs = numEpochs
        self.batch_size = batch_size
        self.lr = lr
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.x_training_steps = x_training_steps
        self.y_training_steps = y_training_steps
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
    def y_training_data(self):
        return torch.linspace(self.y_start, self.y_end, steps=self.y_training_steps)

    @property
    def y_testing_data(self):
        return torch.linspace(self.y_start, self.y_end, steps=self.testing_steps)

    @property
    def training_grid(self):
        return torch.meshgrid(self.x_training_data, self.y_training_data)

    @property
    def testing_grid(self):
        return torch.meshgrid(self.x_testing_data, self.y_testing_data)

    @property
    def training_data(self):
        return [(x_training, y_training) for x_training, y_training in zip(self.training_grid[0].reshape(-1), self.training_grid[1].reshape(-1))]

    @property
    def testing_data(self):
        return [(x_testing, y_testing) for x_testing, y_testing in zip(self.testing_grid[0].reshape(-1), self.testing_grid[1].reshape(-1))]

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

            for batch_index, (x_batch, y_batch) in enumerate(self.__training_loader):

                loss = torch.zeros([1])

                for x, y in zip(x_batch, y_batch):

                    x = x.reshape(-1)
                    y = y.reshape(-1)
                    G_ = self.G(x, y)
                    loss += criterion(G_, torch.tensor([0.]))

                loss /= len(x_batch)
                self.loss_array = np.append(self.loss_array, loss.item())

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                print(
                    f"Epoch: {epoch} - Batch: {batch_index} - Loss:{loss.item():.3e}")

    def psi_trial(self, x, y):
        return (self.psi_hat(x, y) + self.F(x, y) * self.net(torch.tensor([x, y])))

    def dpsiTrial_dx2_dy2(self, x, y):
        x.requires_grad_(True)
        y.requires_grad_(True)

        psi_trial_ = self.psi_trial(x, y)
        grad_x, = torch.autograd.grad(psi_trial_,x,create_graph=True)
        hessian_x, = torch.autograd.grad(grad_x,x,create_graph=True)
        hessian_xy, = torch.autograd.grad(grad_x,y,create_graph=True)

        grad_y, = torch.autograd.grad(psi_trial_,y,create_graph=True)
        hessian_y, = torch.autograd.grad(grad_y,y,create_graph=True)
        hessian_yx, = torch.autograd.grad(grad_y,x,create_graph=True)

        assert torch.abs(hessian_xy-hessian_yx) < 1e-04, "Mixed derivatives must be equal!"

        return hessian_x, hessian_y

    def G(self, x, y):
        psi_trial_ = self.psi_trial(x, y)
        dpsiTrial_dx2_, dpsiTrial_dy2_ = self.dpsiTrial_dx2_dy2(x, y)
        return self.diff_equation(x, y, dpsiTrial_dx2_, dpsiTrial_dy2_, psi_trial_)


    # Exact solution
    def exact_solution(self, x, y):
        if not(self.exact_solution_):
            raise Exception("Exact solution not set yet!")
        return eval(self.exact_solution_)

    # Differential equation
    def diff_equation(self, x, y, dpsi_dx2, dpsi_dy2, psi):
        if not(self.diff_equation_):
            raise Exception("Differential equation not set yet!")
        return eval(self.diff_equation_)

    # Function that constraints the neural network
    def F(self, x, y):
        if not(self.F_):
            raise Exception("Function F not set yet!")
        return eval(self.F_)

    # Function that satisfies the boundary conditions
    def psi_hat(self, x, y):
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

        from utilities.save_results_2D import save_results_elliptic
        from utilities.plot_results_2D import plot_results_elliptic

        X =  self.testing_grid[0]
        Y =  self.testing_grid[1]
        nn_solution = torch.zeros_like(X)
        for row in range(nn_solution.shape[0]):
            for column in range(nn_solution.shape[1]):
                nn_solution[row, column] = self.psi_trial(X[row, column], Y[row, column])
        exact_solution = self.exact_solution(X, Y)
        error = np.abs(exact_solution - nn_solution.detach().numpy())
        training_loss = self.loss_array

        save_results_elliptic(self.filename, X, Y,  nn_solution.detach().numpy(), exact_solution, error, training_loss)
        plot_results_elliptic(self.filename)


    