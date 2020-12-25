import os
import itertools
import torch
from termcolor import colored
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import shutil
import numpy as np
import chart


__data_dir = "data"
__daily_charts_dir = "data\\daily_charts"
__intraday_charts_dir = "data\\intraday_charts"

__fundamentals_dir = "data\\fundamentals"
__fundamentals_file = "data\\fundamentals.csv"

__active_days_file = "data\\active_days.csv"

__datasets_dir = "data\\datasets"
__log_dir = "log"


__labels_dir = "labels"
__generated_labels_dir = __labels_dir + "\\automatic"
__manual_labels_dir = __labels_dir + "\\manual"
__production_label_dir = "data\\production_labels"


def get_production_labels_for(symbol, date):
    path = get_production_label_path_for(symbol, date)
    labels = None
    if os.path.isfile(path):
        labels = np.load(path)
    else:
        print(colored("ERROR! Label file not found at: " + path, color="red"))

    return labels


def get_generated_label_path_for(symbol, date):
    date = date.replace("-", "")
    return __generated_labels_dir + "\\" + symbol + "_" + str(date) + ".npy"


def get_manual_label_path_for(symbol, date):
    date = date.replace("-", "")
    return __manual_labels_dir + "\\" + symbol + "_" + str(date) + ".npy"


def get_production_label_path_for(symbol, date):
    date = date.replace("-", "")
    return __production_label_dir + "\\" + symbol + "_" + str(date) + ".npy"


def load_labels(symbol, date):
    date = date.replace("-", "")
    labels = None
    label_type = None

    path = get_manual_label_path_for(symbol, date)
    if os.path.isfile(path):
        print(colored("Manual labels loaded  ->  " + path, color='green'))
        labels = np.load(path)
        label_type = 1
    else:
        print(colored("No manual labeling found: " + path, color='yellow'))
        path = get_generated_label_path_for(symbol, date)
        if os.path.isfile(path):
            print("Automatic labels loaded  -> ", path)
            labels = np.load(path)
            label_type = 0
        else:
            print(colored("ERROR! No label for: " + symbol + " " + date, color="red"))

    return labels, label_type


def get_fundamentals():

    res = None

    if os.path.isfile(__fundamentals_file):
        res = pd.read_csv(__fundamentals_file)
    else:
        print(colored("File not found:", __fundamentals_file, color='red'))
    return res


def get_fundamentals_for(df, symbol):

    res = None

    x_idx = df.index[df['sym'] == symbol].tolist()
    if len(x_idx) > 0:
        x_idx = x_idx[0]
        res = df.loc[x_idx]

    return res


def stats(gains, show_in_rows=False, show_header=True, show_chart_=False):
    plus = sum(x > 0 for x in gains)
    splus = sum(x for x in gains if x > 0)
    minus = sum(x < 0 for x in gains)
    sminus = sum(x for x in gains if x < 0)

    num_trades = len(gains)
    success_rate = None if (plus + minus) == 0 else plus / (plus + minus)
    rr = None if plus == 0 or minus == 0 else -(splus / plus) / (sminus / minus)

    avg_win = None if plus == 0 else splus / plus
    avg_loss = None if minus == 0 else sminus / minus

    if avg_win is None:
        win_value = 0
    else:
        win_value = num_trades * success_rate * avg_win

    if avg_loss is None:
        loss_value = 0
    else:
        loss_value = num_trades * (1 - success_rate) * avg_loss

    profit = win_value + loss_value

    if len(gains) > 0:
        if show_in_rows:
            if show_header:
                print("Nr.Trades    Success Rate    R/R     Winners    Avg. Win     Losers     Avg. Loss      Profit/Trade")

            print("    ", end="")
            print(num_trades, end="")
            print("           ", end="")
            print(round(100 * success_rate), "%", end="")
            print("        ", end="")
            print("N/A" if rr is None else "%.2f" % rr, end="")
            print("        ", end="")
            print(plus, end="")
            print("        ", end="")
            print("N/A" if avg_win is None else "%.2f" % avg_win + "%", end="")
            print("        ", end="")
            print(minus, end="")
            print("        ", end="")
            print("N/A" if avg_loss is None else "%.2f" % avg_loss + "%", end="")
            print("          ", end="")
            print("%.2f" % (profit / num_trades))
        else:
            print("")
            print("Nr Trades:", num_trades)
            print("Success Rate:", success_rate, "%")
            print("R/R:", "N/A" if rr is None else "%.2f" % rr)
            print("Winners:", plus, " Avg. Win:", "N/A" if avg_win is None else "%.2f" % avg_win + "%")
            print("Losers:", minus, " Avg. Loss:", "N/A" if avg_loss is None else "%.2f" % avg_loss + "%")
            print("Profit/Trade: %.2f" % (profit / num_trades))
            print("")

    if show_chart_:
        x = list(range(0, len(gains)))
        plt.bar(x, gains)
        plt.show()
        plt.close("all")


def limit(x, mn, mx):
    if x < mn:
        x = mn
    if x > mx:
        x = mx
    return x


def get_time_index(df, date, h, m, s):
    idx = None

    date = str(date).replace("-", "")

    xdate = pd.to_datetime(date, format="%Y%m%d")

    xtime = df.iloc[0]["Time"]
    xtime = xtime.replace(year=xdate.year, month=xdate.month, day=xdate.day, hour=h, minute=m, second=s)

    x_idx = df.index[df['Time'] == xtime].tolist()
    n = len(x_idx)
    if n == 1:
        idx = x_idx[0]
    elif n > 1:
        print(colored("ERROR ... Intraday chart contains more than one bars with same time stamp!!!", color='red'))
    else:
        print(colored("Warning!!! ... Intraday chart contains no timestamp: " + str(xtime) + "   n: " + str(n), color='yellow'))

    return idx


def get_chart_data_prepared_for_ai(symbol, date, p):
    date = str(date).replace("-", "")
    df = get_intraday_chart_for(symbol, date)

    if df is not None:
        idx = get_time_index(df, date, p['__chart_begin_hh'], p['__chart_begin_mm'], 0)
        if idx is not None:
            df = df[idx:]
            df.reset_index(drop=True, inplace=True)

        idx = get_time_index(df, date, p['__chart_end_hh'], p['__chart_end_mm'], 0)
        if idx is not None:
            df = df[:idx+1]
            df.reset_index(drop=True, inplace=True)

    return df


def get_list_of_files(dir_path):
    files = []

    if os.path.isdir(dir_path):
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                files.append(name.replace(".csv", ""))

    return files


def get_chart(path):
    df = None
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        print(colored(path + ' not available.', color='yellow'))
    return df


def make_dir(path):
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)


def erase_dir_if_exists(path):
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except OSError:
            print("Deletion of the directory %s failed" % path)


def get_list_of_symbols_with_daily_chart():
    symbols = []
    for root, dirs, files in os.walk(__daily_charts_dir, topdown=False):
        for f in files:
            symbols.append(f.replace(".csv", ""))

    return symbols


def get_list_of_intraday_chart_files():
    symbols = []
    for root, dirs, files in os.walk(__intraday_charts_dir + "\\", topdown=False):
        for name in files:
            symbols.append(name)

    return symbols


def get_list_of_symbols_with_intraday_chart():
    symbols = []
    for root, dirs, files in os.walk(__intraday_charts_dir + "\\", topdown=False):
        for name in dirs:
            symbols.append(name)

    return symbols


def get_intraday_chart_files_for(symbol):
    file_list = []
    dir_path = __intraday_charts_dir + "\\" + symbol
    if os.path.isdir(dir_path):
        for file in os.listdir(dir_path):
            if file.endswith(".csv"):
                filename = os.path.join(dir_path, file)
                file_list.append(filename)
    return file_list


def get_daily_chart_path_for(symbol):
    return __daily_charts_dir + "\\" + symbol + ".csv"


def get_index_of_day(df, date, symbol_info=""):
    search_date = pd.to_datetime(str(date).replace("-", ""), format="%Y%m%d")
    xdate = df.iloc[0]["Time"]
    xdate = xdate.replace(year=search_date.year, month=search_date.month, day=search_date.day)

    x_idx = df.index[df['Time'] == xdate].tolist()
    n = len(x_idx)

    idx = None

    if n == 1:
        idx = x_idx[0]
    elif n > 1:
        print(colored(symbol_info + "  >>  ERROR ... Daily chart contains more than one bars with same time stamp!!!", color='red'))
    else:
        print(colored(symbol_info + "  >> ERROR!!! ... Daily chart contains no timestamp: " + str(xdate) + "   n: " + str(n), color='red'))

    return idx


def get_period_before(df, date, period_length, symbol_info=""):
    date_idx = get_index_of_day(df, date, symbol_info)
    df_period = None

    if period_length > 0:
        if date_idx is not None:
            start_idx = date_idx - period_length + 1
            if start_idx < 0:
                start_idx = 0

            df_period = df[start_idx:date_idx + 1].copy()
            if len(df_period) == 0:
                df_period = None
                print(colored(symbol_info + " >> Warning! History length is zero for date: " + str(date), color="yellow"))

    return df_period, date_idx


def get_daily_chart_for(symbol):
    path = __daily_charts_dir + "\\" + symbol + ".csv"

    df = None

    if os.path.isfile(path):
        df = pd.read_csv(path)
        df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S")
    else:
        print(colored('Warning! Daily chart for ' + symbol + ' not available', color='yellow'))

    return df


def get_volume_for(symbol, date):
    volume = 0

    df = get_daily_chart_for(symbol)

    if df is not None:
        date = str(date)

        x_idx = df.index[df['Time'] == date].tolist()

        if len(x_idx) > 0:
            x_idx = x_idx[0]
            volume = df['Volume'][x_idx]

    return volume


def get_premarket_volume_for(params):
    df = get_intraday_chart_for(params['symbol'], params['date'])

    vol = None

    if df is not None:
        open_index = get_time_index(df, params['date'], params['__chart_begin_hh'], params['__chart_begin_mm'], 0)

        if open_index is not None:
            vol = sum(df['Volume'][:open_index].tolist())

    return vol


def get_intraday_chart_for(symbol, date):
    date = date.replace("-", "")

    path = __intraday_charts_dir + "\\" + symbol + "\\" + symbol + "_" + date + ".csv"

    df = None
    if os.path.isfile(path):
        df = pd.read_csv(path)
        df.Time = pd.to_datetime(df.Time, format="%Y-%m-%d  %H:%M:%S")
    else:
        print(colored('Warning! Intraday chart for ' + symbol + ' not available', color='yellow'))

    return df


def intraday_chart_exists_for(symbol, date):
    date = date.replace("-", "")

    path = __intraday_charts_dir + "\\" + symbol + "\\" + symbol + "_" + date + ".csv"

    return os.path.isfile(path)


def show_daily_chart(symbol):
    df = get_daily_chart_for(symbol)
    df = df.set_index(pd.Index(df.Time))

    mpf.plot(df, type='candle', style="charles", volume=True)
    plt.show()


def show_intraday_chart(symbol, date):
    df = get_intraday_chart_for(symbol, date)
    df = df.set_index(pd.Index(df.Time))

    mpf.plot(df, type='candle', volume=True, title=symbol + " " + date)

    plt.show()


def save_intraday_data(symbol, filename, df):
    if len(df) > 0:

        dir_path = __intraday_charts_dir + "\\" + symbol
        make_dir(dir_path)

        df.to_csv(filename, index=False)


def database_info():
    print("\nDatabase Info:")
    data = get_list_of_symbols_with_daily_chart()
    print(" - " + str(len(data)) + " Daily charts")
    data = get_list_of_intraday_chart_files()
    print(" - " + str(len(data)) + " Intraday charts")
    data = get_list_of_symbols_with_intraday_chart()
    print(" - " + str(len(data)) + " stocks with Intraday chart samples")


def normalize_middle(x):
    range_ = 0

    if len(x) > 1:
        mn = min(x)
        mx = max(x)
        range_ = mx - mn

        if range_ > 0:
            x = 2 * (x - (mn + range_ / 2))
        else:
            range_ = abs(mx)

    elif len(x) == 1:
        range_ = abs(x[0])

    if range_ > 0:
        x = x / range_

    return x


def normalize_0_1(x):
    range_ = 0

    if len(x) > 1:
        mn = min(x)
        mx = max(x)
        range_ = mx - mn

        if range_ > 0:
            x = x - mn
        else:
            range_ = mx

    elif len(x) == 1:
        range_ = x[0]

    if range_ != 0:
        x = x / range_

    return x


def scale_to_1(x):
    if len(x) > 0:
        mx = max(x)

        if mx > 0:
            x = x / mx

    return x


def shift_and_scale(x, scale_factor=5, bias=15):
    x = np.array(x)
    x *= scale_factor
    x = x + [bias] * len(x)

    return x


def merge(path1, path2, result_path):
    byte_data1 = np.load(path1)
    byte_data2 = np.load(path2)

    byte_data = np.concatenate((byte_data1, byte_data2))

    np.save(result_path, byte_data)


def analyze_dataset_balance(dataset_path, num_classes):
    print(colored("Loading Data From:" + dataset_path + " ...", color="green"))

    chart_data = np.load(dataset_path)

    labels = []

    n = len(chart_data)
    for i in range(n):
        target = int(chart_data[i][-1])
        labels.append(target)

    print("Dataset Size:", n, "      Data Size:", chart.EXT_DATA_SIZE)
    hist, w = calc_rebalancing_weigths(labels, num_classes)
    print("Dataset Class Histogram:", hist)
    print("Dataset Re-balancing Weights:", w)

    plt.hist(labels, bins=num_classes, alpha=0.5, align='mid', rwidth=4)
    plt.show()


def analyze_ext_dataset_balance(dataset_path, num_classes):
    print(colored("Loading Data From:" + dataset_path + " ...", color="green"))

    chart_data = np.load(dataset_path)
    labels = []

    n = len(chart_data)
    for i in range(n):
        target = int(chart_data[i][-1])

        labels.append(target)

    print("Dataset Size:", n, "      Data Size:", chart.EXT_DATA_SIZE)
    hist, w = calc_rebalancing_weigths(labels, num_classes)
    print("Dataset Class Histogram:", hist)
    print("Dataset Re-balancing Weights:", w)

    plt.hist(labels, bins=num_classes, alpha=0.5, align='mid', rwidth=4)
    plt.show()


def calc_rebalancing_weigths(y, num_classes):
    hist, _ = np.histogram(y, bins=num_classes)
    avg = sum(hist) / num_classes

    w = []
    for i in range(0, num_classes):
        w.append(avg / hist[i])

    return hist, w


def analyze_divided_dataset_balance(dataset_path, num_dataset_chunks, num_classes):
    labels = []
    num_data = 0

    for dataset_idx in range(num_dataset_chunks):
        path = dataset_path + "_" + str(dataset_idx) + ".npy"
        print(colored("Loading Data From:" + path + " ...", color="green"))

        chart_data = np.load(path)

        n = len(chart_data)
        for i in range(n):
            target = int(chart_data[i][-1])
            labels.append(target)

        num_data += n
        print("Dataset[%s] Size: %s     Data Size: %s" % (dataset_idx, n, chart.EXT_DATA_SIZE))

    print("")
    print("Dataset Size:", num_data, "      Data Size:", chart.EXT_DATA_SIZE)
    hist, w = calc_rebalancing_weigths(labels, num_classes)
    print("Dataset Class Histogram:", hist)
    print("Dataset Re-balancing Weights:", w)

    plt.hist(labels, bins=num_classes, alpha=0.5, align='mid', rwidth=4)
    plt.show()


def show_1min_chart_normalized(df, info, save_to_dir=None, filename=None):
    df = df.set_index(pd.Index(df.Time))
    title = str(info)

    if save_to_dir is None:  # Display chart
        mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume', volume=True, figscale=1, figratio=[16, 9], title=title)
    else:  # Save Chart in file
        make_dir(save_to_dir)
        if filename is None:
            path = save_to_dir + "\\" + "----" + ".png"
        else:
            path = save_to_dir + "\\" + filename + ".png"

        mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume', savefig=path, volume=True, figscale=2, figratio=[16, 9], title=title)


def print_confusion_matrix(confusion, num_classes):
    print("Confusion Matrix:")

    print("[", end="")
    for i in range(num_classes):
        print("[", end="")
        for j in range(num_classes):
            print("%s" % confusion[i][j], end="")
            if j < num_classes-1:
                print(", ", end="")
        print("]", end="")
        if i < num_classes - 1:
            print(",")
        else:
            print("]")


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def calc_accuracy_from_confusion_matrix(cm, num_classes):
    accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        s = 0
        for k in range(num_classes):
            s += cm[k][i]
        if s > 0:
            accuracy[i] = 100 * cm[i][i] / s

    return accuracy


def progress_points(i, at_step, max_line_length=100):
    if i % at_step == 0 and i > 1:
        print(".", end="")
    if i % (at_step * max_line_length) == 0 and i > 1:
        print("")


def random_split(input_samples_path, out1_path, out2_path, split_coefficient, seed):
    input_samples = np.load(input_samples_path)

    num_input_samples = len(input_samples)

    out1_size = int(num_input_samples * split_coefficient)
    out2_size = num_input_samples - out1_size

    print("Number Of Samples: ", num_input_samples)
    print("Out1 Size: ", out1_size, "(%.1f" % (100 * out1_size / num_input_samples), "%)")
    print("Out2 Size: ", out2_size, "(%.1f" % (100 * out2_size / num_input_samples), "%)")

    print("Generating Random Split")
    out1_idx, out2_idx = torch.utils.data.random_split(
        range(num_input_samples),
        [out1_size, out2_size],
        generator=torch.Generator().manual_seed(seed)
    )

    write_indexed_samples_to_file(input_samples, out1_idx, out1_path)
    write_indexed_samples_to_file(input_samples, out2_idx, out2_path)


def write_indexed_samples_to_file(samples, indexes, path):
    indexed_samples = []
    n = len(indexes)
    for i in range(0, n):
        indexed_samples.append(samples[indexes[i]])
    out_data = np.array(indexed_samples)
    np.save(path, out_data)


def calc_range(min_price, max_price):
    if min_price <= 0:
        min_price = 0.1
    return 100*(max_price / min_price - 1)