import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, output_classes):
        super(Net, self).__init__()

        self.dropout = nn.Dropout(0.0001)

        input_size = 2560
        self.fc1 = nn.Linear(input_size, 10000)
        self.fc2 = nn.Linear(10000, 5000)
        self.fc3 = nn.Linear(5000, 5000)
        self.fc4 = nn.Linear(5000, output_classes)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = F.relu(self.fc4(x))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)
