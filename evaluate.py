import os
import torch
import numpy as np
from termcolor import colored
from torchsummary import summary
import time
from model import Net
import chart
import common as cu


def test(model, device, test_loader, p):
    print("Testing Model ")

    start_time = time.time()

    model.eval()

    confusion = np.zeros((p['num_classes'], p['num_classes']))

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)

            prediction = output.argmax(dim=1, keepdim=True)

            ################ CONFUSION MATRIX ###############################
            tgt = target.view_as(prediction)
            prd = prediction

            n = len(prd)
            for k in range(n):
                confusion[tgt[k][0]][prd[k][0]] += 1
            #################################################################

            if i % 10 == 0:
                print(".", end="")
            if i % 1000 == 0 and i > 1:
                print("")

    duration = time.time() - start_time
    print('\nCompleted in %.2f sec' % duration)

    accuracy = cu.calc_accuracy_from_confusion_matrix(confusion, p['num_classes'])

    return accuracy, confusion


def load_data(p):
    dataset_path = p['dataset_path']

    print(colored("Loading Data From:" + dataset_path + " ...", color="green"))

    chart_data = np.load(dataset_path)

    num = len(chart_data)

    dataset = []
    labels = []

    print("Dataset Size:", num)

    for i in range(num):
        state = chart_data[i][:-1]
        state = np.reshape(state, (chart.DATA_ROWS, chart.EXTENDED_CHART_LENGTH))
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
    summary(model, input_size=(1, chart.DATA_ROWS, chart.EXTENDED_CHART_LENGTH))

    path = p['model_params_file_path']

    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
        print(colored("Loaded AI state file: " + path, color="green"))

        test_loader, labels = load_data(p)
        accuracy, cm = test(model, device, test_loader, p)

        print("Accuracy: ", end="")
        for i in range(p['num_classes']):
            print("%.1f" % accuracy[i] + str("%"), " ", end="")
        print("")

        cu.print_confusion_matrix(cm, p['num_classes'])
        cu.plot_confusion_matrix(np.array(cm), np.array(range(p['num_classes'])), normalize=True)
        cu.plot_confusion_matrix(np.array(cm), np.array(range(p['num_classes'])), normalize=False)

    else:
        print(colored("Could not find AI state file: " + path, color="red"))


def get_params():
    params = {
        'num_classes': 2,
        'test_batch': 1024,
        'model_params_file_path': 'checkpoints\\checkpoint_3',
        'dataset_path': 'data\\datasets\\test_data.npy',
    }

    return params


if __name__ == '__main__':
    main()
