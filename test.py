import pandas as pd
import numpy as np
from termcolor import colored
import chart
import common as cu
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mplfinance as mpf
import torch.nn.functional as F
import torch.nn as nn
from array import array


def test6():
    x = [0, 5, 12, 0, 1, -3]
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = []
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = [0, 5, 5, 5, 5, 10]
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = [5, 5, 5, 5, 5, 5]
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = [5]
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = [0]
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = [-5, -5]
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = [-5, -15, 0, 0, 0, -20]
    print(x, " => ", cu.normalize_middle(np.array(x)))


def test7():
    x = [0, 5, 12, 0, 1, -3]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = []
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = [0, 5, 5, 5, 5, 10]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = [5, 5, 5, 5, 5, 5]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = [5]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = [0]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = [-5, -5]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = [-5, -15, 0, 0, 0, -20]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    z = -0.65
    print(z, " ----> ", cu.shift_and_scale([z]), "\n")
    z = [-0.65]
    print(z, " ----> ", cu.shift_and_scale(z), "\n")
    z = [0]
    print(z, " ----> ", cu.shift_and_scale(z), "\n")
    z = [1]
    print(z, " ----> ", cu.shift_and_scale(z), "\n")
    z = [-1]
    print(z, " ----> ", cu.shift_and_scale(z), "\n")


def test8():
    x = [0, 0, 0, 0, 0, 0]
    print(x, " => ", cu.normalize_0_1(np.array(x)))

    x = [0, 5, 12, 0, 1, -3]
    print(x, " => ", cu.normalize_0_1(np.array(x)))

    x = []
    print(x, " => ", cu.normalize_0_1(np.array(x)))

    x = [0, 5, 5, 5, 5, 10]
    print(x, " => ", cu.normalize_0_1(np.array(x)))

    x = [5, 5, 5, 5, 5, 5]
    print(x, " => ", cu.normalize_0_1(np.array(x)))

    x = [5]
    print(x, " => ", cu.normalize_0_1(np.array(x)))

    x = [0]
    print(x, " => ", cu.normalize_0_1(np.array(x)))

    x = [-5, -5]
    print(x, " => ", cu.normalize_0_1(np.array(x)))

    x = [-5, -15, 0, 0, 0, -20]
    print(x, " => ", cu.normalize_0_1(np.array(x)))


def test_dot_prod():
    a = np.array([1, 2, 3, 4, 5, 6])
    b = np.array([2, 0, 2, 0, 2, 0])

    c = a * b
    print(c)

    for i in range(12):
        print(i)


def test9():
    x = [0, 5, 12, 0, 1, -3]
    print(x, " => ", cu.scale_to_1(np.array(x)))

    x = []
    print(x, " => ", cu.scale_to_1(np.array(x)))

    x = [0, 5, 5, 5, 5, 10]
    print(x, " => ", cu.scale_to_1(np.array(x)))

    x = [5, 5, 5, 5, 5, 5]
    print(x, " => ", cu.scale_to_1(np.array(x)))

    x = [5]
    print(x, " => ", cu.scale_to_1(np.array(x)))

    x = [0]
    print(x, " => ", cu.scale_to_1(np.array(x)))

    x = [-5, -5]
    print(x, " => ", cu.scale_to_1(np.array(x)))

    x = [-5, -15, 0, 0, 0, -20]
    print(x, " => ", cu.scale_to_1(np.array(x)))

    x = [15, 15, 17, 13, 14, 20]
    print(x, " => ", cu.scale_to_1(np.array(x)))


def test_split():
    x, y = torch.utils.data.random_split(range(30), [20, 10], generator=torch.Generator().manual_seed(42))

    for batch_idx, (data) in enumerate(x):
        print(data, " ", end="")
    print()
    for batch_idx, (data) in enumerate(y):
        print(data, " ", end="")
    print()

    train_kwargs = {'batch_size': 5}
    test_kwargs = {'batch_size': 5}
    cuda_kwargs = {'shuffle': True}

    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(x, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(y, **test_kwargs)

    for batch_idx, (data) in enumerate(train_loader):
        print(data, " ", end="")
    print()
    for batch_idx, (data) in enumerate(train_loader):
        print(data, " ", end="")
    print("     -> ", len(x))

    for batch_idx, (data) in enumerate(test_loader):
        print(data, " ", end="")
    print()
    for batch_idx, (data) in enumerate(test_loader):
        print(data, " ", end="")
    print("     -> ", len(y))


def test_stratified_sampler():
    dataset = torch.from_numpy(np.array([2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 2, 2, 0, 0, 2, 0, 2, 0, 1, 0, 2, 2, 0]))
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        shuffle=True,
        stratify=dataset)

    print(train_idx)
    print(test_idx)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=3, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=3, sampler=test_sampler)

    for data in train_loader:
        print(data)


def test_rebalance_weights(dataset_path):
    print(colored("Loading Data From:" + dataset_path + " ...", color="green"))
    float_data = np.load(dataset_path)

    chart_size = chart.DATA_ROWS * chart.DAY_IN_MINUTES
    label_size = 1
    data_size = chart_size + label_size

    num_bytes = len(float_data)
    num_rows = int(num_bytes / data_size)

    chart_data = float_data.reshape(num_rows, data_size)
    labels = []

    print("Dataset Size:", num_rows, "      Data Size:", data_size)

    for i in range(num_rows):
        target = int(chart_data[i][-1])

        labels.append(target)

        if i % 5000 == 0:
            print(".", end="")

    print("")
    hist, w = cu.calc_rebalancing_weigths(labels, num_classes=4)
    print("Dataset Class Histogram:", hist)
    print("Dataset Re-balancing Weights:", w)
    w = torch.tensor(w, dtype=torch.float)
    print(w)


def merge():
    cu.merge('data\\datasets\\dev_test_data',
             'data\\datasets\\test_data',
             'data\\datasets\\xxxxx')


def plot_matrix():
    cm = [
        [2053.0, 1588.0, 835.0, 789.0, 2463.0, 3466.0, 3186.0],
        [2105.0, 3387.0, 5146.0, 3436.0, 11906.0, 8497.0, 4479.0],
        [946.0, 2816.0, 10810.0, 14212.0, 21474.0, 7826.0, 2540.0],
        [620.0, 1926.0, 14158.0, 167060.0, 37343.0, 5189.0, 2070.0],
        [3362.0, 5503.0, 14482.0, 32059.0, 42464.0, 17854.0, 7577.0],
        [2669.0, 3691.0, 6365.0, 5358.0, 19302.0, 16046.0, 6351.0],
        [4198.0, 3068.0, 2125.0, 2698.0, 7160.0, 9425.0, 9077.0]
    ]

    cu.plot_confusion_matrix(np.array(cm), np.array(range(7)), normalize=True)
    cu.print_confusion_matrix(cm, 7)


def test_dataset():
    dataset_path = 'data\\datasets\\training_data_0.npy'

    print(colored("Loading Data From:" + dataset_path + " ...", color="green"))

    chart_data = np.load(dataset_path)

    print("Dataset Size:", len(chart_data), "      Data Size:", chart.EXT_DATA_SIZE)

    p = {
        '__chart_begin_hh': 9,
        '__chart_begin_mm': 30,
        '__chart_end_hh': 15,
        '__chart_end_mm': 59,
    }
    symbol = "AAL"
    date = "2020-02-26"
    df = cu.get_intraday_chart_for(symbol, date)
    t = df.Time.to_list()
    idx = 0
    while idx < 5000:
        chart.save_state_chart(chart_data[idx][:-1], t, "---", "___", idx, chart.EXTENDED_CHART_LENGTH)
        idx += 20


def pad_to512():
    dataset_path = 'data\\datasets\\dataset_x'

    print(colored("Loading Data From:" + dataset_path + " ...", color="green"))

    float_data = np.load(dataset_path)

    chart_size_bytes = chart.DATA_ROWS * chart.DAY_IN_MINUTES
    label_size_bytes = 1
    data_size = chart_size_bytes + label_size_bytes

    num_bytes = len(float_data)
    num_rows = int(num_bytes / data_size)

    chart_data = float_data.reshape(num_rows, data_size)
    print("Dataset Size:", num_rows, "      Data Size:", data_size)

    s = chart_data[0][:-1]
    x = chart.pad_state_to_512(s)
    print(x.shape)
    print(x)


def test_dynamic_candle():
    symbol = "CGC"
    date = "2018-08-16"

    params = {
        '__chart_begin_hh': 9,
        '__chart_begin_mm': 30,
        '__chart_end_hh': 15,
        '__chart_end_mm': 59,

        'trading_begin_hh': 9,
        'trading_begin_mm': 40,
        'symbol': symbol,
        'date': date
    }

    print(cu.get_premarket_volume_for(params))

    cu.show_daily_chart(symbol)

    period_length = 7
    df = cu.get_daily_chart_for(params['symbol'])
    xxx, idx = cu.get_period_before(df, params['date'], period_length)
    print(df["Time"][idx])
    print(df["Time"][idx - period_length + 1:idx + 1])

    ####################################################################
    params['Open'] = df['Close'][idx]
    params['Close'] = df['Close'][idx]
    params['High'] = df['High'][idx]
    params['Low'] = df['Low'][idx]
    params['Volume'] = float(df['Volume'][idx])*0.01

    params['date_index'] = idx
    ####################################################################

    xxx = chart.update_candle(xxx, params)
    xxx = xxx.set_index(pd.Index(xxx.Time))
    mpf.plot(xxx, type='candle', style="charles", volume=True)

    ################ TEST IF ORIGINAL DF WAS MODIFIED ##################
    idx += 1
    xxx = df[idx - period_length + 1:idx + 1].copy()
    # xxx = chart.update_candle(xxx, idx, params)
    xxx = xxx.set_index(pd.Index(xxx.Time))
    mpf.plot(xxx, type='candle', style="charles", volume=True)
    ###################################################################

    cu.show_intraday_chart(symbol, date)


def test_accuracy_cm():
    cm = [[1.0, 40.0, 0.0, 0.0, 0.0, 7.0, 0.0],
          [51.0, 37.0, 45.0, 1.0, 78.0, 33.0, 0.0],
          [0.0, 3.0, 15.0, 50.0, 2.0, 27.0, 0.0],
          [0.0, 0.0, 10.0, 104.0, 1.0, 0.0, 0.0],
          [8.0, 53.0, 24.0, 82.0, 100.0, 25.0, 0.0],
          [3.0, 33.0, 3.0, 0.0, 26.0, 74.0, 2.0],
          [3.0, 32.0, 0.0, 0.0, 1.0, 25.0, 1.0]]

    cm = np.array(cm)
    acc = cu.calc_accuracy_from_confusion_matrix(cm, 7)

    for i in range(7):
        print("%.2f" % (100*acc[i]), " ", end="")
    print("")

    x = []
    x.append(acc)
    acc = acc * 1.1
    x.append(acc)
    acc = acc * 2
    x.append(acc)
    acc = acc * 5
    x.append(acc)

    x = np.array(x)
    print(x)

    x = x.T
    print("")

    print(x[0])

    for i in range(7):
        plt.plot(x[i])
    plt.show()


def test_xxx():
    n = 4
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(x[:n])
    print(x[n:])


def test_nll_loss():

    w = [1/5.3589, 1/2.1937, 1/1.3094, 1/0.3621, 1/0.6153, 1/1.3060, 1/2.2640]
    print("WWWWW", sum(w))

    # input is of size N x C = 3 x 5
    x = torch.zeros(3, 5, requires_grad=False)
    x[0][0] = 1
    x[1][1] = 1
    x[2][2] = 1
    print(x, "\n")

    y = torch.tensor([3, 3, 3])

    print(F.log_softmax(x, dim=1), "y =>", y)

    print("Loss:")
    w = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float)
    z3 = F.nll_loss(F.log_softmax(x, dim=1), y, reduction='none', weight=w)
    z33 = F.nll_loss(F.log_softmax(x, dim=1), y, reduction='mean', weight=w)
    print(z3, "Mean:", z33)
    w = torch.tensor([1, 1, 1, 2, 1], dtype=torch.float)
    z3 = F.nll_loss(F.log_softmax(x, dim=1), y, reduction='none', weight=w)
    z33 = F.nll_loss(F.log_softmax(x, dim=1), y, reduction='mean', weight=w)
    print(z3, "Mean:", z33)
    print()


def test_rebalance_weights_1():
    x = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    h, w = cu.calc_rebalancing_weigths_INVERSE(x, 3)

    print(h)
    print(w)


def test_w():
    x = np.array([5.3572, 2.1931, 1.0335, 0.3307, 0.6982, 1.6349, 3.1002])
    s = sum(x)
    y = 7 * (x / s)
    print(y)
    print(sum(y))


def test_write_file():

    x = np.asarray([3114, 2.7, 0.0, -1.0, 1.1, 5000.0001])
    np.save("data\\test.npy", x)
    y = np.load("data\\test.npy")
    print(y)


# test_w()

# test_xxx()
# test8()

# test_dynamic_candle()

# test_write_file()

# test_dataset()

# test_nll_loss()

# test_stratified_sampler()
# cu.analyze_ext_dataset_balance('data\\datasets\\x1.npy', num_classes=2)

# cu.analyze_divided_dataset_balance('data\\datasets\\training_data', 11, num_classes=2)

# test6()

# test_split()
# merge()
'''
cu.random_split(input_samples_path="data\\datasets\\training_data_2.npy",
                out1_path="data\\datasets\\x1",
                out2_path="data\\datasets\\x2",
                split_coefficient=0.38,
                seed=9)
'''
# test_rebalance_weights('data\\winner_datasets_2\\winner_dataset_4')

# test_rebalance_weights_1()
# test_split()
