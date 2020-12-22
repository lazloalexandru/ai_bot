import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, output_classes):
        super(Net, self).__init__()

        self.dropout = nn.Dropout(0.05)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=2)

        k = 3

        f = 64
        self.conv1 = nn.Conv2d(1, f, kernel_size=(2, k), stride=1)
        self.bn1 = nn.BatchNorm2d(f)
        self.conv2 = nn.Conv2d(f, f, kernel_size=(1, k), stride=1)
        self.bn2 = nn.BatchNorm2d(f)

        f_prev = f
        f = f * 2
        self.conv3 = nn.Conv2d(f_prev, f, kernel_size=(1, k), stride=1)
        self.bn3 = nn.BatchNorm2d(f)
        self.conv4 = nn.Conv2d(f, f, kernel_size=(1, k), stride=1)
        self.bn4 = nn.BatchNorm2d(f)

        linear_input_size = f * 94
        d = 256

        print("Dense Layers %s / %s / %s / %s" % (linear_input_size, d, d, output_classes))

        self.fc1 = nn.Linear(linear_input_size, d)
        self.fc2 = nn.Linear(d, d)
        self.fc3 = nn.Linear(d, output_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.max_pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.max_pool(x)
        x = self.dropout(x)

        # print(x.shape)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
