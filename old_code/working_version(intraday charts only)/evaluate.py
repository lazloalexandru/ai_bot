import os
import torch
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
from torchsummary import summary
import time
import matplotlib.pyplot as plt
from model_conv import Net
import chart
import common as cu


def test(model, device, test_loader, p):
    print("Testing Model ")

    start_time = time.time()

    model.eval()
    correct = 0

    confusion = np.zeros((p['num_classes'], p['num_classes']))

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            tgt = target.view_as(pred)
            prd = pred

            n = len(prd)
            for k in range(n):
                confusion[tgt[k][0]][prd[k][0]] += 1

            if i % 10 == 0:
                print(".", end="")
            if i % 1000 == 0 and i > 1:
                print("")

    duration = time.time() - start_time
    print('\nCompleted in %.2f sec' % duration)

    accuracy = 100.0 * correct / len(test_loader.dataset)

    return accuracy, confusion


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

        if i % 1000 == 0 and i > 1:
            print(".", end="")
        if i % 100000 == 0 and i > 1:
            print("")

    print("")

    test_kwargs = {'batch_size': p['test_batch']}
    cuda_kwargs = {'pin_memory': True, 'shuffle': True}

    test_kwargs.update(cuda_kwargs)

    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

    return test_loader, labels


def main():
    p = get_params()

    device = torch.device("cuda")

    model = Net(p['num_classes']).to(device)
    summary(model, input_size=(1, chart.DATA_ROWS, chart.DAY_IN_MINUTES))

    path = p['model_params_file_path']

    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
        print(colored("Loaded AI state file: " + path, color="green"))

        test_loader, labels = load_data(p)
        accuracy, cm = test(model, device, test_loader, p)

        print("Accuracy: %.2f%s" % (accuracy, "%"))
        cu.print_confusion_matrix(cm, p['num_classes'])
        cu.plot_confusion_matrix(np.array(cm), np.array(range(p['num_classes'])), normalize=p['normalized_confusion'])

    else:
        print(colored("Could not find AI state file: " + path, color="red"))


def get_params():
    params = {
        'num_classes': 7,
        'test_batch': 1024,
        'model_params_file_path': 'checkpoints\\checkpoint_339',
        'dataset_path': 'data\\datasets\\test_dataset',
        'normalized_confusion': False
    }

    return params


if __name__ == '__main__':
    main()
