import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, output_classes):
        super(Net, self).__init__()

        self.dropout = nn.Dropout(0.1)

        input_size = 2560
        d = 2000

        self.fc1 = nn.Linear(input_size, d)
        self.fc2 = nn.Linear(d, d)
        self.fc3 = nn.Linear(d, d)
        self.fc4 = nn.Linear(d, d)
        self.fc5 = nn.Linear(d, d)
        self.fc6 = nn.Linear(d, d)
        self.fc7 = nn.Linear(d, d)
        self.fc8 = nn.Linear(d, output_classes)

        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)
