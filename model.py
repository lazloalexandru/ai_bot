import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        f = 128
        self.conv1 = nn.Conv2d(1, f, kernel_size=(5, 3), stride=1)
        self.bn1 = nn.BatchNorm2d(f)
        self.conv2 = nn.Conv2d(f, f, kernel_size=(1, 3), stride=1)
        self.bn2 = nn.BatchNorm2d(f)

        linear_input_size = 49408
        print("Dense Layer %s x %s" % (linear_input_size, 2))

        self.fc1 = nn.Linear(linear_input_size, 2048)
        self.fc2 = nn.Linear(2048, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
