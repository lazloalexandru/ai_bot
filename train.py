from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from termcolor import colored
from model import Net

import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def plot_values(values):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(values, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Profit')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


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
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        accuracy))

    return accuracy


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


def get_params():
    params = {
        'train_batch': 5000,
        'test_batch': 5000,

        'num_epochs': 1000,
        'save_epoch_step': 100,
        'resume_idx': 200,
        'dataset_path': 'data\\winner_dataset.dat'
    }

    return params


def main():
    plt.ion()

    p = get_params()

    device = torch.device("cuda")

    train_kwargs = {'batch_size': p['train_batch']}
    test_kwargs = {'batch_size': p['test_batch']}

    dataset_path = p['dataset_path']
    dataset1 = load_data(dataset_path, training_set=True)
    dataset2 = load_data(dataset_path, training_set=False)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    resume_idx = p['resume_idx']

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

    accuracy_history = []

    for epoch in range(start_idx + 1, start_idx + num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)

        accuracy_history.append(accuracy)
        plot_values(accuracy_history)
        
        if epoch % 100 == 0:
            torch.save(model.state_dict(), "checkpoints\\checkpoint_" + str(epoch))

    print(colored('Training Complete!', color="green"))

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
