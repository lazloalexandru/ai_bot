import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        linear_input_size = 128 * 189
        outputs = 6
        print("Dense Layers %s ... %s" % (linear_input_size, outputs))

        self.conv1 = nn.Conv2d(1, 256, kernel_size=(5, 7), stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(1, 7), stride=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        self.dropout1 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(linear_input_size, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.max_pool(x)
        x = self.dropout1(x)

        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def get_params():
    params = {
        'num_classes': 6,

        'training_batch_log_interval': 10,
        'train_batch': 512,
        'test_batch': 512,

        'loss_ceiling': 3,

        'resume_epoch_idx': None,
        'num_epochs': 10000,
        'checkpoint_at_epoch_step': 20,

        'seed': 92,
        'data_reload_counter_start': 0,
        'dataset_path': 'data\\datasets\\dataset',
        'dataset_chunks': 4,
        'split_coefficient': 0.8,
        'change_dataset_at_epoch_step': 150
    }

    return params
