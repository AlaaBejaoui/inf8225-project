import torch
from NeuralNetSolver import NeuralNetSolver
from timing import time_it

torch.manual_seed(0)

solver = NeuralNetSolver(numHiddenLayer=0, numUnits=10, activation="leakyReLU", 
                        numEpochs=500, batch_size=5, lr=0.03, 
                        x_start=0, x_end=2, y_start=0, y_end=2, training_steps=5, 
                        testing_steps=3, 
                        shuffle=False) ##chabe true


print(solver.training_data)

for batch_index, (batch1,batch2) in enumerate(solver.training_loader): #chage to __trainig_loader
    print(batch1,batch2)