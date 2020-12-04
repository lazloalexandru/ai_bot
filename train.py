from __future__ import print_function
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from termcolor import colored
from torch.optim.lr_scheduler import StepLR
from model import Net


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    log_interval = 1

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output.float(), target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = F.nll_loss(output.float(), target)
            test_loss += loss  # F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def load_data(dataset_path, training_set=True):
    byte_data = np.fromfile(dataset_path, dtype='float')

    num_bytes = len(byte_data)
    rows = int(num_bytes / 1951)

    chart_data = byte_data.reshape(rows, 1951)
    data = []

    print(colored("Loading Data From:" + dataset_path, color="green"))
    print("Dataset Size:", rows)

    if training_set:
        start_idx = 1
        end_idx = int(rows*0.8) + 1
    else:
        start_idx = int(rows * 0.8) + 2
        end_idx = rows

    for i in range(start_idx, end_idx):
        state = chart_data[i][:-1]
        state = np.reshape(state, (5, 390))
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to("cuda")

        target = int(chart_data[i][-1])

        data.append((state, target))

        if i % 1000 == 0:
            print(".", end="")

    print("")

    return data


def main():
    # Training settings

    device = torch.device("cuda")

    train_kwargs = {'batch_size': 2000}
    test_kwargs = {'batch_size': 500}

    dataset_path = 'data\\sell_dataset.dat'
    dataset1 = load_data(dataset_path, training_set=True)
    dataset2 = load_data(dataset_path, training_set=False)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    resume_idx = 200

    if resume_idx is not None:
        path = "checkpoints\\checkpoint_" + str(resume_idx)
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path))
            print(colored("Loaded AI state file: " + path, color="green"))
        else:
            print(colored("Could not find AI state file: " + path, color="red"))
            resume_idx = None

    optimizer = optim.Adadelta(model.parameters(), lr=1)

    num_epochs = 1000
    if resume_idx is None:
        start_idx = 1
    else:
        start_idx = resume_idx

    for epoch in range(start_idx + 1, start_idx + num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

        if epoch % 100 == 0:
            torch.save(model.state_dict(), "checkpoints\\checkpoint_" + str(epoch))


if __name__ == '__main__':
    main()
