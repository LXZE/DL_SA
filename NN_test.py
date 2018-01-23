import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.conv_l1 = nn.Conv2d(1,6,5)
        self.conv_l2 = nn.Conv2d(6,12,5)

        # 12 matrix size 5*5 feed to fc
        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)

    def forward(self, x):
        
        x = F.max_pool2d(F.relu(self.conv_l1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv_l2(x)), (2, 2))

        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        n_feature = 1
        for s in size:
            n_feature *= s
        return n_feature

net = NN()
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)

# net.zero_grad()
# out.backward(torch.randn(1, 10))

optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()

output = net(input)
target = Variable(torch.arange(1, 11))
loss = criterion(output, target)
loss.backward()
optimizer.step()