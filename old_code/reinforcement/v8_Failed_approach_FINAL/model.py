import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        linear_input_size = h * w
        # print("Dense Layer %s x %s" % (linear_input_size, outputs))

        self.fc1 = nn.Linear(linear_input_size, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.fc4 = nn.Linear(1000, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x.view(x.size(0), -1))
