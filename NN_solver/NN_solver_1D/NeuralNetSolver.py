import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd())
from NN_solver.NN_solver_1D.neuralNetwork import Net
from utilities.timing import time_it


class NeuralNetSolver:

    def __init__(self, numHiddenLayer, numUnits, activation, numEpochs, batch_size, lr, x_start, x_end, training_steps, testing_steps, shuffle, filename):
        self.net = Net(numHiddenLayer, numUnits, activation)

        self.numEpochs = numEpochs
        self.batch_size = batch_size
        self.lr = lr
        self.x_start = x_start
        self.x_end = x_end
        self.training_steps = training_steps
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
    def training_data(self):
        return torch.linspace(self.x_start, self.x_end, steps=self.training_steps)

    @property
    def testing_data(self):
        return torch.linspace(self.x_start, self.x_end, steps=self.testing_steps)

    @property
    def __training_loader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, shuffle=self.shuffle)

    @property
    def __testing_loader(self):
        return DataLoader(self.testing_data, batch_size=1)

    @time_it
    def solve(self):
        criterion = nn.MSELoss()

        optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        # optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

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
                    f"Epoch: {epoch} - Batch: {batch_index} - Loss:{loss.item():.3e}")

    def psi_trial(self, x):
        return (self.psi_hat(x) + self.F(x) * self.net(x))

    def dpsiTrial_dx(self, x):
        psi_trial_ = self.psi_trial(x)
        psi_trial_.backward(create_graph=True)
        return x.grad

    def G(self, x):
        psi_trial_ = self.psi_trial(x)
        dpsiTrial_dx_ = self.dpsiTrial_dx(x)
        return self.diff_equation(x, dpsiTrial_dx_, psi_trial_)

    # Exact solution
    def exact_solution(self, x):
        if not(self.exact_solution_):
            raise Exception("Exact solution not set yet!")
        return eval(self.exact_solution_)

    # Differential equation 
    def diff_equation(self, x, dpsi_dx, psi):
        if not(self.diff_equation_):
            raise Exception("Differential equation not set yet!")
        return eval(self.diff_equation_)

    # Function that constraints the neural network
    def F(self, x):
        if not(self.F_):
            raise Exception("Function F not set yet!")
        return eval(self.F_) 

    # Function that satisfies the boundary conditions
    def psi_hat(self, x):
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

        from utilities.save_results_1D import save_results
        from utilities.plot_results_1D import plot_results

        for x_test in self.__testing_loader:
            self.psi_trial_array = np.append(
                self.psi_trial_array, self.psi_trial(x_test).item())

        x =  self.testing_data
        nn_solution = self.psi_trial_array
        exact_solution = self.exact_solution(x)
        error = np.abs(self.exact_solution(x) - nn_solution)
        training_loss = self.loss_array

        save_results(self.filename, x, nn_solution, exact_solution, error, training_loss)
        plot_results(self.filename)

    
