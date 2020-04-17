import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from neuralNetwork import Net

torch.manual_seed(0)

numEpochs = 500
batch_size = 5
lr = 0.01

x_start = 0
x_end = 1

y_start = 0
y_end = 1

training_steps = 20
testing_steps = 100

x_training_data = torch.linspace(x_start, x_end, steps=training_steps)
x_testing_data = torch.linspace(x_start, x_end, steps=testing_steps)

y_training_data = torch.linspace(y_start, y_end, steps=training_steps)
y_testing_data = torch.linspace(y_start, y_end, steps=testing_steps)

x_grid_training, y_grid_training = torch.meshgrid(x_training_data, y_training_data)
x_grid_testing, y_grid_testing = torch.meshgrid(x_testing_data, y_testing_data)

training_data = [(x_training,y_training) for x_training, y_training in zip(x_grid_training.reshape(-1),y_grid_training.reshape(-1))]
testing_data = [(x_testing,y_testing) for x_testing, y_testing in zip(x_grid_testing.reshape(-1),y_grid_testing.reshape(-1))]

# print(training_data)
# print()
# print(testing_data)

# raise ArithmeticError

training_loader = DataLoader(training_data, batch_size= batch_size, shuffle=True)
testing_loader = DataLoader(testing_data, batch_size= 1)

net = Net(numHiddenLayer=2, numUnits=10, activation="leakyReLU")

loss_array = np.array([])
psi_trial_array = np.array([])

#exact solution  
def exact_solution(x, y):
    return ( x*(y-1) + x*x*np.exp(-2*y) + np.exp(-x*x*np.exp(-2*y)) + x*np.exp(-y) )

#differential equation to solve 
def diff_equation(x, y, dpsi_dx, dpsi_dy, psi):
    return (x*dpsi_dx + dpsi_dy - x*y)

#function that constraints the neural network
def F(x, y):
    return y

#function that satisfies the boundary conditions  
def psi_hat(x, y):
    return (torch.pow(x, 2) + torch.exp(-torch.pow(x, 2)))

def psi_trial(x, y):
    return (psi_hat(x, y) + F(x, y) * net(torch.tensor([x, y])))

def dpsiTrial_dx(x, y):
    psi_trial_ = psi_trial(x, y)
    psi_trial_.backward() 
    return x.grad

def dpsiTrial_dy(x, y):
    psi_trial_ = psi_trial(x, y)
    psi_trial_.backward() 
    return y.grad

def G(x, y):
    psi_trial_ = psi_trial(x, y)
    dpsiTrial_dx_ = dpsiTrial_dx(x, y)
    dpsiTrial_dy_ = dpsiTrial_dy(x, y)
    return diff_equation(x, y, dpsiTrial_dx_, dpsiTrial_dy_, psi_trial_)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr)

for epoch in range(numEpochs):
    
    for batch_index, (x_batch, y_batch) in enumerate(training_loader):

        loss = torch.zeros([1])
        for x, y in zip(x_batch, y_batch):

            x.resize_((1,1))
            y.resize_((1,1))
            x.requires_grad_(True)
            y.requires_grad_(True)
            G_ =  G(x, y)
            loss += criterion(G_, torch.tensor([0.]))
        loss /= len(x_batch)
        loss_array = np.append(loss_array, loss.item())
        
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch} - batch: {batch_index} - loss:{loss.item():.3e}")

#solution of the NN
for x_test in testing_loader:
    psi_trial_array = np.append(psi_trial_array, psi_trial(x_test).item())

#comparing the two solutions
plt.figure()
plt.semilogy(loss_array)
plt.xlabel("iter")
plt.ylabel("loss")

#comparing the two solutions
plt.figure()
plt.plot(testing_data, psi_trial_array, 'r', label='NN')
plt.xlabel("x")
plt.ylabel("y(x)")
plt.plot(testing_data, exact_solution(testing_data), 'k', label='exact')
plt.legend()

#error plot
plt.figure()
err = exact_solution(testing_data) - psi_trial_array
plt.plot(testing_data, np.abs(err), 'k')
plt.xlabel("x")
plt.ylabel("|err(x)|")
plt.show()