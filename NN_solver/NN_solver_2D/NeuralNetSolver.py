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
        optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, nesterov=True)
        
        for epoch in range(self.numEpochs):

            for batch_index, (x_batch, y_batch) in enumerate(self.__training_loader):
                # print(x_batch,y_batch)
                loss = torch.zeros([1])
                for x, y in zip(x_batch, y_batch):
                    x.resize_((1, 1))
                    y.resize_((1, 1))
                    x.requires_grad_(True)
                    y.requires_grad_(True)
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

    def dpsiTrial_dx(self, x, y):
        if x.grad != None and y.grad != None:
            x.grad.data.zero_()
            y.grad.data.zero_()
        psi_trial_ = self.psi_trial(x, y)
        psi_trial_.backward()
        return x.grad

    def dpsiTrial_dy(self, x, y):
        if x.grad != None and y.grad != None:
            x.grad.data.zero_()
            y.grad.data.zero_()
        psi_trial_ = self.psi_trial(x, y)
        psi_trial_.backward()
        return y.grad

    def G(self, x, y):
        psi_trial_ = self.psi_trial(x, y)
        dpsiTrial_dx_ = self.dpsiTrial_dx(x, y)
        dpsiTrial_dy_ = self.dpsiTrial_dy(x, y)
        return self.diff_equation(x, y, dpsiTrial_dx_, dpsiTrial_dy_, psi_trial_)

    def plot(self):
    
        # NN solution
        fig = plt.figure(figsize=(4, 4))

        Z_NN = torch.zeros_like(self.testing_grid[0])
        for line in range(Z_NN.shape[0]):
            for column in range(Z_NN.shape[1]):
                # print(self.testing_grid[0][line, column])
                # print()
                # print(self.testing_grid[1][line, column])
                Z_NN[line, column] = self.psi_trial(
                    self.testing_grid[0][line, column], self.testing_grid[1][line, column])

        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        cp = ax.contour(self.testing_grid[0].detach().numpy(
        ), self.testing_grid[1].detach().numpy(), Z_NN.detach().numpy(), 20)
        ax.clabel(cp, inline=True,
                  fontsize=10)
        ax.set_title('Contour Plot of the NN solution')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        # exact solution contour plot
        fig = plt.figure(figsize=(4, 4))
        Z_exact = self.exact_solution(
            self.testing_grid[0], self.testing_grid[1])
        left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
        ax = fig.add_axes([left, bottom, width, height])
        cp = ax.contour(self.testing_grid[0],
                        self.testing_grid[1], Z_exact, 20)
        ax.clabel(cp, inline=True,
                  fontsize=10)
        ax.set_title('Contour Plot of the exact solution')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
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
