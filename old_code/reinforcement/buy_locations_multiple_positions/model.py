import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        f = 150
        self.conv1 = nn.Conv2d(1, f, kernel_size=(7, 3), stride=1)
        self.bn1 = nn.BatchNorm2d(f)
        self.conv2 = nn.Conv2d(f, f, kernel_size=(1, 3), stride=1)
        self.bn2 = nn.BatchNorm2d(f)
        self.conv3 = nn.Conv2d(f, f, kernel_size=(1, 3), stride=1)
        self.bn3 = nn.BatchNorm2d(f)
        self.conv4 = nn.Conv2d(f, f, kernel_size=(1, 3), stride=1)
        self.bn4 = nn.BatchNorm2d(f)
        self.conv5 = nn.Conv2d(f, f, kernel_size=(1, 3), stride=1)
        self.bn5 = nn.BatchNorm2d(f)

        def conv2d_size_out(size, kernel_size=3):
            return size - kernel_size + 1

        convw = conv2d_size_out(w)
        convw = conv2d_size_out(convw)
        convw = conv2d_size_out(convw)
        convw = conv2d_size_out(convw)
        convw = conv2d_size_out(convw)
        convh = 1

        linear_input_size = convw * convh * f
        print("Dense Layer %s x %s" % (linear_input_size, outputs))

        self.dl1 = nn.Linear(linear_input_size, 5000)
        self.dl2 = nn.Linear(5000, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.dl1(x.view(x.size(0), -1)))
        return self.dl2(x.view(x.size(0), -1))
