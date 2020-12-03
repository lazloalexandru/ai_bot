import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1950, 5000)
        self.bn1 = nn.BatchNorm1d(5000)
        self.fc2 = nn.Linear(5000, 5000)
        self.bn2 = nn.BatchNorm1d(5000)
        self.fc3 = nn.Linear(5000, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        print(output.shape)
        print(output)
        return output
