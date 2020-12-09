from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import numpy as np
from termcolor import colored
from model import Net
from model import get_params
import matplotlib
import matplotlib.pyplot as plt
import chart


def test(model, device, test_loader):
    print("Testing Model ")

    model.eval()
    correct = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            print(".", end="")

            if i % 101 == 0:
                print("")

    print("")

    accuracy = 100.0 * correct / len(test_loader.dataset)

    return accuracy


def load_data(p):
    dataset_path = p['dataset_path']

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

    print("Dataset Size:", num_rows, "      Data Size:", data_size)

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

    print("")

    test_kwargs = {'batch_size': p['test_batch']}
    cuda_kwargs = {'pin_memory': True, 'shuffle': True}

    test_kwargs.update(cuda_kwargs)

    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

    return test_loader, labels


def main():
    plt.ion()

    p = get_params()

    device = torch.device("cuda")

    model = Net().to(device)

    path = p['model_params_file_path']

    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
        print(colored("Loaded AI state file: " + path, color="green"))

        test_loader, labels = load_data(p)

        accuracy = test(model, device, test_loader)

        print("Accuracy:", accuracy)
    else:
        print(colored("Could not find AI state file: " + path, color="red"))


def get_params():
    params = {
        'num_classes': 6,
        'test_batch': 512,
        'model_params_file_path': 'checkpoints\\checkpoint_700',
        'dataset_path': 'data\\datasets\\dataset_4'
    }

    return params


if __name__ == '__main__':
    main()