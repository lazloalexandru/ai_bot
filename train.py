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
import chart
import gc
import common as cu
import time

__global_iteration_counter = 0
__iteration_loss_history = []


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


def train(model, device, train_loader, optimizer, epoch, w, p):
    global __global_iteration_counter

    start_time = time.time()

    model.train()

    losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        __global_iteration_counter += 1

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output.float(), target, weight=w)

        losses.append(loss.item())
        if p['log_iteration_loss']:
            __iteration_loss_history.append(loss.item())

        loss.backward()
        optimizer.step()

        if batch_idx % p['training_batch_log_interval'] == 0:
            print('{}. Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                __global_iteration_counter, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), sum(losses) / len(losses)))

    duration = time.time() - start_time
    print('%.2f sec' % (duration))

    return sum(losses) / len(losses)


def test(model, device, test_loader, w):
    start_time = time.time()
    model.eval()
    losses = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            losses.append(F.nll_loss(output, target, weight=w).item())

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = sum(losses) / len(losses)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset), accuracy))

    duration = time.time() - start_time
    print('%.2f sec' % (duration))

    return accuracy, avg_loss


def load_data(p):
    dataset_path = get_dataset_path(p)

    print(colored("Loading Data From:" + dataset_path + " ...", color="green"))

    float_data = np.fromfile(dataset_path, dtype='float')

    chart_size_bytes = chart.DATA_ROWS * chart.DAY_IN_MINUTES
    label_size_bytes = 1
    data_size = chart_size_bytes + label_size_bytes

    num_bytes = len(float_data)
    num_rows = int(num_bytes / data_size)

    chart_data = float_data.reshape(num_rows, data_size)
    dataset = []
    labels = []

    print("Dataset Size:", num_rows, "      Data Size:", data_size,  "   <-   Seed:", p['seed'])

    training_set_size = int(num_rows * p['split_coefficient'])
    test_set_size = num_rows - training_set_size

    print("Training Dataset Size:", training_set_size)
    print("Test Dataset Size:", test_set_size)

    for i in range(num_rows):
        state = chart_data[i][:-1]
        state = np.reshape(state, (chart.DATA_ROWS, chart.DAY_IN_MINUTES))
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        target = int(chart_data[i][-1])
        target = torch.tensor(target)

        dataset.append((state, target))
        labels.append(target)

        if i % 10000 == 0:
            print(".", end="")
        if i % 1000000 == 0 and i > 1:
            print("")

    print("")
    print("Balancing / Splitting Data / Resampling ...")

    #####################################################################################################
    # Calculate Re-Balancing Weights

    hist, w = cu.calc_rebalancing_weigths(labels, p['num_classes'])
    print("Dataset Class Histogram:", hist)
    print("Dataset Re-balancing Weights:", w)
    w = torch.tensor(w, dtype=torch.float).to("cuda")

    #####################################################################################################
    #  RANDOM SPLIT

    training_data, test_data = torch.utils.data.random_split(
        dataset, [training_set_size, test_set_size], generator=torch.Generator().manual_seed(p['seed'])
    )

    train_kwargs = {'batch_size': p['train_batch']}
    test_kwargs = {'batch_size': p['test_batch']}
    cuda_kwargs = {'pin_memory': True, 'shuffle': True}

    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(training_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)
    '''
    #####################################################################################################
    #  STRATIFIED SPLIT
    
    from sklearn.model_selection import train_test_split

    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        shuffle=True,
        stratify=labels)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_kwargs = {'batch_size': p['train_batch'], 'sampler': train_sampler}
    test_kwargs = {'batch_size': p['test_batch'], 'sampler': test_sampler}
    cuda_kwargs = {'pin_memory': True, 'shuffle': False} # Stratified sample does shuffle 
    
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    
    train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)
    '''

    return train_loader, test_loader, w


def get_dataset_path(p):
    if p['dataset_chunks'] > 1:
        dataset_path = p['dataset_path'] + "_" + str(p['dataset_id'])
    else:
        dataset_path = p['dataset_path']

    return dataset_path


def save_loss_history(p):
    cu.make_dir(p['loss_history_files_dir'])

    xxx = np.array(__iteration_loss_history)
    path = p['loss_history_files_dir'] + "\\" + str(p['train_batch']) + "_" + str(p['learning_rate']) + ".dat"
    print("Saving Iterations:", len(__iteration_loss_history), "  ->  ", path)
    xxx.tofile(path)


def main():
    plt.ion()

    p = get_params()

    device = torch.device("cuda")

    model = Net(get_params()['num_classes']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=p['learning_rate'])

    resume_idx = p['resume_epoch_idx']

    if resume_idx is not None:
        path = "checkpoints\\checkpoint_" + str(resume_idx)
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path))
            print(colored("Loaded AI state file: " + path, color="green"))
        else:
            print(colored("Could not find AI state file: " + path, color="red"))
            resume_idx = None

    num_epochs = p['num_epochs']
    if resume_idx is None:
        start_idx = 1
    else:
        start_idx = resume_idx + 1

    accuracy_history = []
    train_losses = []
    test_losses = []

    train_loader = None
    test_loader = None
    w_re_balance = None

    if p['dataset_chunks'] == 1:
        train_loader, test_loader, w_re_balance = load_data(p)

    data_reload_counter = p['data_reload_counter_start']

    for epoch in range(start_idx, start_idx + num_epochs):
        if p['dataset_chunks'] > 1 and p['change_dataset_at_epoch_step'] is not None:
            if epoch % p['change_dataset_at_epoch_step'] == 0 or data_reload_counter == p['data_reload_counter_start']:
                del train_loader
                del test_loader
                torch.cuda.empty_cache()
                gc.collect()

                p['dataset_id'] = data_reload_counter % p['dataset_chunks']
                train_loader, test_loader, w_re_balance = load_data(p)
                data_reload_counter += 1

        if train_loader is not None and test_loader is not None:
            train_loss = train(model, device, train_loader, optimizer, epoch, w_re_balance, p)
            accuracy, test_loss = test(model, device, test_loader, w_re_balance)

            if test_loss > p['loss_ceiling']:
                test_loss = p['loss_ceiling']

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracy_history.append(accuracy)
            plot_values(accuracy_history, train_losses, test_losses)

            if epoch % p['checkpoint_at_epoch_step'] == 0:
                torch.save(model.state_dict(), "checkpoints\\checkpoint_" + str(epoch))
        else:
            print(colored("Train and Test Data Loaders not Initialized!!!", color='red'))
            break

    if p['log_iteration_loss']:
        save_loss_history(p)

    print('Finished!')

    plt.ioff()
    plt.show()


def get_params():
    params = {
        ################# Non Essential #######################
        'loss_ceiling': 3,
        'training_batch_log_interval': 50,

        ################# Learning Rate Analysis ##############
        'log_iteration_loss': False,
        'loss_history_files_dir': None,

        ################# Model ###############################
        'num_classes': 7,

        ################ Training - Dataset ###################
        'seed': 92,
        'data_reload_counter_start': 0,
        'dataset_path': 'data\\datasets\\dataset_01',
        'dataset_chunks': 1,
        'split_coefficient': 0.8,
        'change_dataset_at_epoch_step': 200,

        ################ Training #############################
        'train_batch': 128,
        'test_batch': 2048,
        'learning_rate': 0.0001,

        'num_epochs': 200,
        'checkpoint_at_epoch_step': 5,
        'resume_epoch_idx': 135
    }

    return params


if __name__ == '__main__':
    main()
