import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(1, 10)  
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

x = torch.tensor([1.], requires_grad=True)

out = net(x)

out.backward()

print("input: ", x)
print("output: ", out)
print("grad: ", x.grad)

x = torch.tensor([1.1], requires_grad=True)

out = net(x)

print("input: ", x)
print("output: ", out)
