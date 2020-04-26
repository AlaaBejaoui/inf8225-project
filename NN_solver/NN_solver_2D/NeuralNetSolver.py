import os
import sys
sys.path.append(os.getcwd())
from utilities.timing import time_it
from NN_solver.NN_solver_2D.neuralNetwork import Net
from matplotlib import ticker, cm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch


class NeuralNetSolver:

    def __init__(self, numHiddenLayer, numUnits, activation, numEpochs, batch_size, lr, x_start, x_end, y_start, y_end, training_steps, testing_steps, shuffle):
        self.net = Net(numHiddenLayer, numUnits, activation)

        self.numEpochs = numEpochs
        self.batch_size = batch_size
        self.lr = lr
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.training_steps = training_steps
        self.testing_steps = testing_steps
        self.shuffle = shuffle

        self.loss_array = np.array([])
        self.psi_trial_array = np.array([])

        self.diff_equation_ = None
        self.F_ = None
        self.exact_solution_ = None
        self.psi_hat_ = None

    @property
    def x_training_data(self):
        return torch.linspace(self.x_start, self.x_end, steps=self.training_steps)

    @property
    def x_testing_data(self):
        return torch.linspace(self.x_start, self.x_end, steps=self.testing_steps)

    @property
    def y_training_data(self):
        return torch.linspace(self.y_start, self.y_end, steps=self.training_steps)

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
                    f"epoch: {epoch} - batch: {batch_index} - loss:{loss.item():.3e}")

    def psi_trial(self, x, y):
        return (self.psi_hat(x, y) + self.F(x, y) * self.net(torch.tensor([x, y])))

    def dpsiTrial_dx_dy(self, x, y):
        x.requires_grad_(True)
        y.requires_grad_(True)
        psi_trial_ = self.psi_trial(x, y)
        psi_trial_.backward(create_graph=True)
        grad_x = x.grad
        grad_y = y.grad
        return grad_x, grad_y

    def G(self, x, y):
        psi_trial_ = self.psi_trial(x, y)
        dpsiTrial_dx_, dpsiTrial_dy_ = self.dpsiTrial_dx_dy(x, y)
        return self.diff_equation(x, y, dpsiTrial_dx_, dpsiTrial_dy_, psi_trial_)

    def plot(self):

        # training loss logplot
        plt.figure()
        plt.title("training loss")
        plt.xlabel("iter")
        plt.ylabel("loss")
        plt.semilogy(self.loss_array)

        # NN solution
        fig = plt.figure()
        Z_NN = torch.zeros_like(self.testing_grid[0])
        for row in range(Z_NN.shape[0]):
            for column in range(Z_NN.shape[1]):
                Z_NN[row, column] = self.psi_trial(
                    self.testing_grid[0][row, column], self.testing_grid[1][row, column])
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        cp = ax.contourf(
            self.testing_grid[0], self.testing_grid[1], Z_NN.detach().numpy(), 40)
        ax.set_title('Contour Plot of the NN solution')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        cbar = fig.colorbar(cp)

        # exact solution contour plot
        fig = plt.figure()
        Z_exact = self.exact_solution(
            self.testing_grid[0], self.testing_grid[1])
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        cp = ax.contourf(self.testing_grid[0],
                         self.testing_grid[1], Z_exact, 40)
        ax.set_title('Contour Plot of the exact solution')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        cbar = fig.colorbar(cp)

        # error
        fig = plt.figure()
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        err = Z_NN - Z_exact
        cp = ax.contourf(self.testing_grid[0], self.testing_grid[1], np.abs(
            err.detach().numpy()), 40)
        ax.set_title('Contour Plot of the error')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        cbar = fig.colorbar(cp)
        plt.show()

    # exact solution
    def exact_solution(self, x, y):
        if not(self.exact_solution_):
            raise Exception("exact solution not set yet!")
        return eval(self.exact_solution_)

    # differential equation to solve
    def diff_equation(self, x, y, dpsi_dx, dpsi_dy, psi):
        if not(self.diff_equation_):
            raise Exception("differential equation not set yet!")
        return eval(self.diff_equation_)

    # function that constraints the neural network
    def F(self, x, y):
        if not(self.F_):
            raise Exception("function F not set yet!")
        return eval(self.F_)

    # function that satisfies the boundary conditions
    def psi_hat(self, x, y):
        if not(self.psi_hat_):
            raise Exception("function psi hat not set yet!")
        return eval(self.psi_hat_)

    def set_exact_solution(self, command):
        self.exact_solution_ = command

    def set_diff_equation(self, command):
        self.diff_equation_ = command

    def set_F(self, command):
        self.F_ = command

    def set_psi_hat(self, command):
        self.psi_hat_ = command
