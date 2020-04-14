import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from neuralNetwork import Net

torch.manual_seed(0)

numEpochs = 50
batch_size = 1
lr = 0.03

x_start = 0
x_end = 1

training_steps = 100
testing_steps = 1000

training_data = torch.linspace(x_start, x_end, steps=training_steps)
testing_data = torch.linspace(x_start, x_end, steps=testing_steps)

training_loader = DataLoader(training_data, batch_size= batch_size)
testing_loader = DataLoader(testing_data, batch_size= 1)

net = Net()

#exact solution  
def exact_solution(x):
    return (0.5 * (np.exp(-x) + np.cos(x) + np.sin(x)))

#function that constraints the neural network
def F(x):
    return x

#function that satisfies the boundary conditions  
def psi_hat(x):
    return 1

def psi_trial(x):
    return (psi_hat(x) + F(x) * net(x))

def dpsiTrial_dx(x):
    psi_trial_ = psi_trial(x)
    psi_trial_.backward() 
    return x.grad

def G(x):
    psi_trial_ = psi_trial(x)
    dpsiTrial_dx_ = dpsiTrial_dx(x)
    return (dpsiTrial_dx_ +  psi_trial_ - torch.cos(x))


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr)

for epoch in range(numEpochs):
    for index, x in enumerate(training_loader):

        #x = torch.tensor([i], requires_grad=True) ###
        x.requires_grad_(True)
        
        G_ =  G(x)
        loss = criterion(G_, torch.tensor([0.]))
        optimizer.zero_grad()   
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch} - index: {index} - loss:{loss}")

#solution of the NN
psi_trial_array = np.array([])
for x_test in testing_loader:
    psi_trial_array = np.append(psi_trial_array, psi_trial(x_test).item())

#comparing the two solutions
plt.plot(testing_data, psi_trial_array, 'r', label='NN')
plt.plot(testing_data, exact_solution(testing_data), 'k', label='exact')
plt.legend()
plt.show()




