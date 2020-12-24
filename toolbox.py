import pandas as pd
import numpy as np
from termcolor import colored
import chart
import common as cu
import torch
import matplotlib.pyplot as plt
import mplfinance as mpf
import torch.nn.functional as F


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
    dataset_path = 'data\\datasets\\test_data.npy'

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
    while idx < 1000:
        chart.save_state_chart(chart_data[idx][:-1], chart_data[idx][-1], t, idx)
        idx += 20


# test_dataset()

# cu.analyze_ext_dataset_balance('data\\datasets\\x1.npy', num_classes=2)
cu.analyze_divided_dataset_balance('data\\datasets\\test_data', 5, num_classes=2)
# merge()
'''
cu.random_split(input_samples_path="data\\datasets\\training_data_3.npy",
                out1_path="data\\datasets\\x1",
                out2_path="data\\datasets\\x2",
                split_coefficient=0.1,
                seed=9)
'''