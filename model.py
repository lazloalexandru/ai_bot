import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(5, 3), stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(1, 3), stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=(1, 3), stride=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout1 = nn.Dropout(0.25)

        linear_input_size = 49152
        outputs = 7
        print("Dense Layer %s x %s" % (linear_input_size, outputs))

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout1(x)
        # print("XXXXXXXXXXXXXXX: ", x.shape)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
