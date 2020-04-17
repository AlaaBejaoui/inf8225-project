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
lr = 0.03

x_start = 0
x_end = 2

training_steps = 20
testing_steps = 10000

training_data = torch.linspace(x_start, x_end, steps=training_steps)
testing_data = torch.linspace(x_start, x_end, steps=testing_steps)

training_loader = DataLoader(training_data, batch_size= batch_size, shuffle=True)
testing_loader = DataLoader(testing_data, batch_size= 1)

net = Net(numHiddenLayer=0, numUnits=30, activation="leakyReLU")

loss_array = np.array([])
psi_trial_array = np.array([])

#exact solution  
def exact_solution(x):
    return (np.exp(-0.2 * x) * np.sin(x))

#differential equation to solve 
def diff_equation(x, dpsi_dx, psi):
    return (dpsi_dx + 0.2 * psi - torch.exp(-0.2 * x) * torch.cos(x))

#function that constraints the neural network
def F(x):
    return x

#function that satisfies the boundary conditions  
def psi_hat(x):
    return 0

def psi_trial(x):
    return (psi_hat(x) + F(x) * net(x))

def dpsiTrial_dx(x):
    psi_trial_ = psi_trial(x)
    psi_trial_.backward() 
    return x.grad

def G(x):
    psi_trial_ = psi_trial(x)
    dpsiTrial_dx_ = dpsiTrial_dx(x)
    return diff_equation(x, dpsiTrial_dx_, psi_trial_)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr)

for epoch in range(numEpochs):
    
    for batch_index, batch in enumerate(training_loader):
        
        loss = torch.zeros([1])
        for x in batch:
            x.resize_((1,1))
            x.requires_grad_(True)
            G_ =  G(x)
            loss += criterion(G_, torch.tensor([0.]))
        loss /= len(batch)
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