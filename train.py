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
from dataset_generator import get_marker

__global_iteration_counter = 0
__iteration_loss_history = []


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def plot_values(accuracy, train_loss, test_loss, p):
    fig = plt.figure(1, figsize=(10, 9))
    plt.clf()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    ax1.plot(train_loss, color='tab:orange', label="train")
    ax1.plot(test_loss, color='tab:red', label="test")
    ax1.legend(loc="upper left")

    accuracy = np.array(accuracy).T

    ax2.plot(accuracy[p['num_classes']], color="white", label="all")
    for i in range(p['num_classes']):
        _, c = get_marker(i)
        ax2.plot(accuracy[i], color=c, label=str(i))

    ax2.legend(loc="upper left")
    ax2.set_facecolor('silver')

    fig.tight_layout()

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def train(model, device, train_loader, optimizer, epoch, w, p):
    global __global_iteration_counter

    epoch_start_time = time.time()

    model.train()

    losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        batch_start_time = time.time()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output.float(), target, weight=w, reduction='mean')

        losses.append(loss.item())
        if p['log_iteration_loss']:
            __iteration_loss_history.append(loss.item())

        loss.backward()
        optimizer.step()

        batch_duration = (time.time() - batch_start_time) * 1000
        if batch_idx % p['training_batch_log_interval'] == 0:
            print('{}. Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}   {:.0f} ms'.format(
                __global_iteration_counter, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), sum(losses) / len(losses), batch_duration))

        __global_iteration_counter += 1

    epoch_duration = time.time() - epoch_start_time
    print('Epoch completed in %.2f sec' % epoch_duration)

    return sum(losses) / len(losses)


def test(model, device, test_loader, w, p):
    start_time = time.time()

    model.eval()

    losses = []
    num_hits = 0
    counter = 0

    confusion = np.zeros((p['num_classes'], p['num_classes']))

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            losses.append(F.nll_loss(output, target, weight=w).item())

            prediction = output.argmax(dim=1, keepdim=True)

            tgt = target.view_as(prediction)
            prd = prediction

            #################################################################
            num_hits += prediction.eq(tgt).sum().item()

            ################ CONFUSION MATRIX ###############################
            n = len(prd)
            for k in range(n):
                confusion[tgt[k][0]][prd[k][0]] += 1

            #################################################################

            cu.progress_points(counter, 10)
            counter += 1

    avg_loss = sum(losses) / len(losses)
    accuracy = cu.calc_accuracy_from_confusion_matrix(confusion, p['num_classes'])
    overall_accuracy = 100.0 * num_hits / len(test_loader.dataset)

    accuracy = np.concatenate((accuracy, [overall_accuracy]))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{}'.format(
        avg_loss, num_hits, len(test_loader.dataset)), " Classes: ", end="")

    for i in range(p['num_classes']):
        print("%.1f" % accuracy[i] + str("%"), " ", end="")
    print("")

    duration = time.time() - start_time
    print('%.2f sec\n' % duration)

    return accuracy, avg_loss


def load_data(dataset_path, batch_size, re_balancing_weights, conv_input_layer):
    start_time = time.time()

    print(colored("Loading Data From:" + dataset_path + " ...", color="green"))
    float_data = np.fromfile(dataset_path, dtype='float')

    data_size = chart.EXT_DATA_SIZE
    num_bytes = len(float_data)
    num_rows = int(num_bytes / data_size)
    chart_data = float_data.reshape(num_rows, data_size)
    dataset = []
    labels = []

    print("Dataset Re-balancing Weights:", re_balancing_weights)
    print("Dataset Size:", num_rows, "      Data Size:", data_size)
    print("Creating Tensors")

    for i in range(num_rows):
        state = chart_data[i][:-1]
        if conv_input_layer:
            state = np.reshape(state, (chart.DATA_ROWS, chart.EXTENDED_CHART_LENGTH))
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        label = int(chart_data[i][-1])
        label = torch.tensor(label)

        dataset.append((state, label))
        labels.append(label)

        cu.progress_points(i, 10000)

    kwargs = {'batch_size': batch_size}
    cuda_kwargs = {'pin_memory': True, 'shuffle': True}
    kwargs.update(cuda_kwargs)

    loader = torch.utils.data.DataLoader(dataset, **kwargs)

    duration = time.time() - start_time
    print(' %.2f sec\n' % duration)

    return loader


def get_training_dataset_path(p):
    if p['dataset_chunks'] > 1:
        dataset_path = p['training_data_path'] + "_" + str(p['dataset_id'])
    else:
        dataset_path = p['training_data_path']

    return dataset_path


def save_loss_history(p):
    cu.make_dir(p['loss_history_files_dir'])

    xxx = np.array(__iteration_loss_history)
    path = p['loss_history_files_dir'] + "\\" + \
        str(p['train_batch']) + "_" + str(p['learning_rate']) + "_" + str(p['weight_decay']) + ".dat"
    print("Saving Iterations:", len(__iteration_loss_history), "  ->  ", path)
    xxx.tofile(path)


def init_iteration_logger(p):
    success = True
    if p['log_iteration_loss']:
        if p['loss_history_files_dir'] is None:
            print(colored("ERROR! \"log_iteration_loss\" parameter set to True, but \"loss_history_files_dir\" is set to None! Specify directory!", color='red'))
            success = False
        if not os.path.isdir(p['loss_history_files_dir']):
            print(colored("ERROR! Cannot find directory: " + p['loss_history_files_dir'], color='red'))
            success = False

    return success


def init_ai(p):
    device = torch.device("cuda")

    model = Net(get_params()['num_classes']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    w_re_balance = torch.tensor(p['re_balancing_weights'], dtype=torch.float).to("cuda")

    resume_idx = p['resume_epoch_idx']

    if resume_idx is not None:
        path = "checkpoints\\checkpoint_" + str(resume_idx)
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path))
            print(colored("Loaded AI state file: " + path, color="green"))
        else:
            print(colored("Could not find AI state file: " + path, color="red"))
            resume_idx = None

    if resume_idx is None:
        start_idx = 1
    else:
        start_idx = resume_idx + 1

    return device, model, optimizer, w_re_balance, start_idx


def main():
    plt.ion()

    p = get_params()

    if not init_iteration_logger(p):
        return

    num_epochs = p['num_epochs']
    device, model, optimizer, w_re_balance, start_idx = init_ai(p)

    accuracy_history = []
    train_losses = []
    test_losses = []

    train_loader = None
    test_loader = load_data(p['dev_test_data_path'],
                            p['test_batch'],
                            p['re_balancing_weights'],
                            p['conv_input_layer'])

    if p['dataset_chunks'] == 1:
        train_loader = load_data(get_training_dataset_path(p),
                                 p['train_batch'],
                                 p['re_balancing_weights'],
                                 p['conv_input_layer'])

    data_reload_counter = p['data_reload_counter_start']

    for epoch in range(start_idx, start_idx + num_epochs):
        if p['dataset_chunks'] > 1 and p['change_dataset_at_epoch_step'] is not None:
            if epoch % p['change_dataset_at_epoch_step'] == 0 or data_reload_counter == p['data_reload_counter_start']:
                del train_loader
                torch.cuda.empty_cache()
                gc.collect()

                p['dataset_id'] = data_reload_counter % p['dataset_chunks']
                train_loader = load_data(get_training_dataset_path(p),
                                         p['train_batch'],
                                         p['re_balancing_weights'],
                                         p['conv_input_layer'])
                data_reload_counter += 1

        if train_loader is not None and test_loader is not None:
            train_loss = train(model, device, train_loader, optimizer, epoch, w_re_balance, p)
            accuracy, test_loss = test(model, device, test_loader, w_re_balance, p)

            if test_loss > p['loss_ceiling']:
                test_loss = p['loss_ceiling']

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracy_history.append(accuracy)
            plot_values(accuracy_history, train_losses, test_losses, p)

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
        'loss_ceiling': 5,
        'training_batch_log_interval': 50,

        ################# Learning Rate Analysis ##############
        'log_iteration_loss': False,
        'loss_history_files_dir': "--",

        ################# Model ###############################
        'num_classes': 5,
        'conv_input_layer': True,

        ################ Dev Test - Data ######################
        'dev_test_data_path': 'data\\datasets\\dev_test_data',
        # 'dev_test_data_path': 'data\\datasets\\dummy',

        ################ Training - Data ######################
        'training_data_path': 'data\\datasets\\training_data',
        # 'training_data_path': 'data\\datasets\\dummy',
        'dataset_chunks': 11,
        're_balancing_weights': [7.1888, 3.0813, 0.2720, 1.8304, 3.1772],

        'data_reload_counter_start': 0,
        'change_dataset_at_epoch_step': 1,
        ################ Training #############################
        'train_batch': 128,
        'test_batch': 512,
        'learning_rate': 0.0001,
        'weight_decay': 0.01,

        'num_epochs': 500,
        'checkpoint_at_epoch_step': 1,
        'resume_epoch_idx': None
    }

    return params


if __name__ == '__main__':
    main()
