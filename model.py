import torch.nn as nn
import torch.nn.functional as F
import torch

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(5, 7), stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(1, 7), stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(1, 7), stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        linear_input_size = 47616
        outputs = 4
        print("Dense Layers %s ... %s" % (linear_input_size, outputs))

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, outputs)

    def forward(self, x):
        # x = self.dropout1(x)
        # x = self.dropout2(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        print("XXXXXX", x.shape)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        linear_input_size = 24192
        outputs = 4
        print("Dense Layers %s ... %s" % (linear_input_size, outputs))

        self.conv1 = nn.Conv2d(1, 128, kernel_size=(5, 7), stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(1, 7), stride=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.dropout1 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(linear_input_size, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # print('XXXXXXXX:', x.shape)

        x = self.max_pool(x)

        # print('XXXXXXXX:', x.shape)

        x = self.dropout1(x)

        # print('XXXXXXXX:', x.shape)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
