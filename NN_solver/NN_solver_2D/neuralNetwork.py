import torch
import torch.nn as nn

class Net(nn.Module):

    def __init__(self, numHiddenLayer=1, numUnits=10, activation="relu"):
        super(Net, self).__init__()
        
        self.inputLayer = nn.Linear(2, numUnits)  
        self.hiddenLayer = nn.Linear(numUnits, numUnits)
        self.outputLayer = nn.Linear(numUnits, 1)
        self.numHiddenLayer = numHiddenLayer
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "sigmoid": 
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError
        


    def forward(self, x):
        
        x = self.inputLayer(x)
        x = self.activation(x)
        for _ in range(self.numHiddenLayer):
            x = self.hiddenLayer(x)
            x = self.activation(x)
        x = self.outputLayer(x)
        return x