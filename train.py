from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from termcolor import colored
from model import Net
import random
import matplotlib
import matplotlib.pyplot as plt
import chart

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_values(accuracy, train_loss, test_loss):
    fig = plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')

    ax1 = plt.gca()
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(accuracy, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.plot(accuracy)

    ax2 = ax1.twinx()

    color = 'tab:orange'
    ax2.set_ylabel('loss', color=color)
    ax2.plot(train_loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    color = 'tab:red'
    ax2.set_ylabel('loss', color=color)
    ax2.plot(test_loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    log_interval = 5

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

    return loss.item()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            '''
            loss = F.nll_loss(output.float(), target)
            test_loss += loss  # F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            '''
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        accuracy))

    return accuracy, test_loss


def load_data(p):
    dataset_path = get_dataset_path(p)

    print(colored("Loading Data From:" + dataset_path + " ...", color="green"))

    float_data = np.fromfile(dataset_path, dtype='float')

    chart_size = chart.DATA_ROWS * chart.DAY_IN_MINUTES
    label_size = 1
    data_size = chart_size + label_size

    num_bytes = len(float_data)
    num_rows = int(num_bytes / data_size)

    chart_data = float_data.reshape(num_rows, data_size)
    data = []

    print("Dataset Size:", num_rows, "      Data Size:", data_size,  "   <-   Seed:", p['seed'])

    split_coefficient = p['split_coefficient']
    training_set_size = int(num_rows * 0.8)
    test_set_size = num_rows - training_set_size

    print("Training Dataset Size:", training_set_size)
    print("Test Dataset Size:", test_set_size)

    for i in range(num_rows):
        state = chart_data[i][:-1]
        state = np.reshape(state, (5, 390))
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to("cuda")

        target = int(chart_data[i][-1])
        target = torch.tensor(target).to("cuda")

        data.append((state, target))

        if i % 5000 == 0:
            print(".", end="")

    print("")

    return data, training_set_size, test_set_size


def get_dataset_path(p):
    if p['dataset_chunks'] > 1:
        dataset_id = random.randint(0, p['dataset_chunks'] - 1)
        dataset_path = p['dataset_path'] + "_" + str(dataset_id)
    else:
        dataset_path = p['dataset_path']

    return dataset_path


def main():
    plt.ion()

    p = get_params()

    device = torch.device("cuda")

    train_kwargs = {'batch_size': p['train_batch']}
    test_kwargs = {'batch_size': p['test_batch']}
    cuda_kwargs = {'shuffle': True}

    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    model = Net().to(device)
    resume_idx = p['resume_epoch_idx']

    if resume_idx is not None:
        path = "checkpoints\\checkpoint_" + str(resume_idx)
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path))
            print(colored("Loaded AI state file: " + path, color="green"))
        else:
            print(colored("Could not find AI state file: " + path, color="red"))
            resume_idx = None

    optimizer = optim.Adadelta(model.parameters(), lr=1)

    num_epochs = p['num_epochs']
    if resume_idx is None:
        start_idx = 1
    else:
        start_idx = resume_idx + 1

    accuracy_history = []
    train_losses = []
    test_losses = []

    reload_data_steps = p['change_dataset_at_epoch_step']

    dataset, train_size, test_size = load_data(p)
    training_data, test_data = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    reload_needed = p['dataset_chunks'] > 1 and reload_data_steps is not None

    for epoch in range(start_idx, start_idx + num_epochs + 1):
        if reload_needed:
            if epoch % reload_data_steps == 0:
                dataset, train_size, test_size = load_data(p)
                training_data, test_data = torch.utils.data.random_split(
                    dataset, [train_size, test_size], generator=torch.Generator().manual_seed(p['seed'])
                )

        if len(training_data) > 0 and len(test_data) > 0:
            train_loader = torch.utils.data.DataLoader(training_data, **train_kwargs)
            test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

            train_loss = train(model, device, train_loader, optimizer, epoch)
            accuracy, test_loss = test(model, device, test_loader)

            if test_loss > p['loss_ceiling']:
                test_loss = p['loss_ceiling']

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracy_history.append(accuracy)
            plot_values(accuracy_history, train_losses, test_losses)

            if epoch % p['checkpoint_at_epoch_step'] == 0:
                torch.save(model.state_dict(), "checkpoints\\checkpoint_" + str(epoch))
        else:
            print(colored("DataSet Too Small!!! Training Data Size: %s    Test Data Size: %s" % (len(dataset1), len(dataset2)), color='red'))

    print(colored('Training Complete!', color="green"))

    plt.ioff()
    plt.show()


def get_params():
    params = {
        'train_batch': 5000,
        'test_batch': 5000,

        'loss_ceiling': 5,

        'resume_epoch_idx': None,
        'num_epochs': 50000,
        'checkpoint_at_epoch_step': 1,

        'seed': 0,
        'dataset_path': 'data\\winner_datasets_2\\winner_dataset_9_10_11',
        'dataset_chunks': 1,
        'split_coefficient': 0.8,
        'change_dataset_at_epoch_step': 200
    }

    return params


if __name__ == '__main__':
    main()
