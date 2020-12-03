import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        '''
        f = 150
        self.conv1 = nn.Conv2d(1, f, kernel_size=(7, 3), stride=1)
        self.bn1 = nn.BatchNorm2d(f)
        self.conv2 = nn.Conv2d(f, f, kernel_size=(1, 3), stride=1)
        self.bn2 = nn.BatchNorm2d(f)
        self.conv3 = nn.Conv2d(f, f, kernel_size=(1, 3), stride=1)
        self.bn3 = nn.BatchNorm2d(f)

        def conv2d_size_out(size, kernel_size=3):
            return size - kernel_size + 1

        convw = conv2d_size_out(w)
        convw = conv2d_size_out(convw)
        convw = conv2d_size_out(convw)
        convh = 1

        linear_input_size = convw * convh * f
        '''
        linear_input_size = 6 * 390 + 1
        print("Dense Layer %s x %s" % (linear_input_size, outputs))

        self.fc1 = nn.Linear(linear_input_size, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x.view(x.size(0), -1))
