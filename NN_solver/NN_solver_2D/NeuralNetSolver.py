import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from neuralNetwork import Net
from timing import time_it


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

        self.diff_equation_ = ""
        self.F_ = ""
        self.exact_solution_ = ""
        self.psi_hat_ = ""

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
        return [(x_training,y_training) for x_training, y_training in zip(self.training_grid[0].reshape(-1),self.training_grid[1].reshape(-1))]

    @property
    def testing_data(self):
        return [(x_testing,y_testing) for x_testing, y_testing in zip(self.testing_grid[0].reshape(-1),self.testing_grid[1].reshape(-1))]

    @property
    def training_loader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, shuffle=self.shuffle)

    @property
    def __testing_loader(self):
        return DataLoader(self.testing_data, batch_size=1)

    @time_it
    def solve(self):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.net.parameters(), lr=self.lr)

        for epoch in range(self.numEpochs):

            for batch_index, batch in enumerate(self.__training_loader):

                loss = torch.zeros([1])
                for x in batch:
                    x.resize_((1, 1))
                    x.requires_grad_(True)
                    G_ = self.G(x)
                    loss += criterion(G_, torch.tensor([0.]))
                loss /= len(batch)
                self.loss_array = np.append(self.loss_array, loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(
                    f"epoch: {epoch} - batch: {batch_index} - loss:{loss.item():.3e}")

    def psi_trial(self, x):
        return (self.psi_hat(x) + self.F(x) * self.net(x))

    def dpsiTrial_dx(self, x):
        psi_trial_ = self.psi_trial(x)
        psi_trial_.backward()
        return x.grad

    def G(self, x):
        psi_trial_ = self.psi_trial(x)
        dpsiTrial_dx_ = self.dpsiTrial_dx(x)
        return self.diff_equation(x, dpsiTrial_dx_, psi_trial_)

    def plot(self):

        for x_test in self.__testing_loader:
            self.psi_trial_array = np.append(
                self.psi_trial_array, self.psi_trial(x_test).item())

        # comparing the two solutions
        plt.figure()
        plt.semilogy(self.loss_array)
        plt.xlabel("iter")
        plt.ylabel("loss")

        # comparing the two solutions
        plt.figure()
        plt.plot(self.testing_data, self.psi_trial_array, 'r', label='NN')
        plt.xlabel("x")
        plt.ylabel("y(x)")
        plt.plot(self.testing_data, self.exact_solution(
            self.testing_data), 'k', label='exact')
        plt.legend()

        # error plot
        plt.figure()
        err = self.exact_solution(self.testing_data) - self.psi_trial_array
        plt.plot(self.testing_data, np.abs(err), 'k')
        plt.xlabel("x")
        plt.ylabel("|err(x)|")

        plt.show()

    # exact solution
    def exact_solution(self, x):
        if not(self.exact_solution_):
            raise Exception("exact solution not set yet!")
        return eval(self.exact_solution_)

    # differential equation to solve
    def diff_equation(self, x, dpsi_dx, psi):
        if not(self.diff_equation_):
            raise Exception("differential equation not set yet!")
        return eval(self.diff_equation_)

    # function that constraints the neural network
    def F(self, x):
        if not(self.F_):
            raise Exception("function F not set yet!")
        return eval(self.F_) 

    # function that satisfies the boundary conditions
    def psi_hat(self, x):
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
