import torch
import torch.nn as nn
import torch.nn.functional as Fun

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # an affine operation: y = Wx + b
        self.inputLayer = nn.Linear(1, 10)  
        self.hiddenLayer = nn.Linear(10, 10)
        self.outputLayer = nn.Linear(10, 1)

    def forward(self, x):
        
        x = Fun.relu(self.inputLayer(x))
        x = Fun.relu(self.hiddenLayer(x))
        x = Fun.relu(self.hiddenLayer(x))
        x = Fun.relu(self.hiddenLayer(x))
        x = self.outputLayer(x)
        return x