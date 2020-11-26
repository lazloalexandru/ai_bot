import os
from termcolor import colored
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import datetime
import shutil
import numpy as np


__data_dir = "\\data"
__daily_charts_dir = "data\\daily_charts"
__intraday_charts_dir = "data\\intraday_charts"
__fundamentals_dir = "data\\fundamentals"
__active_days_file = "data\\active_days.csv"

__normalized_states_root_dir = "normalized_states"
__normalized_states_dir = __normalized_states_root_dir + "\\intraday_charts"


def stats(gains):
    plus = sum(x > 0 for x in gains)
    splus = sum(x for x in gains if x > 0)
    minus = sum(x < 0 for x in gains)
    sminus = sum(x for x in gains if x < 0)

    num_trades = len(gains)
    success_rate = None if (plus + minus) == 0 else round(100 * (plus / (plus + minus)))
    rr = None if plus == 0 or minus == 0 else -(splus / plus) / (sminus / minus)

    avg_win = None if plus == 0 else splus / plus
    avg_loss = None if minus == 0 else sminus / minus

    if len(gains) > 0:
        print("")
        print("Nr Trades:", num_trades)
        print("Success Rate:", success_rate, "%")
        print("R/R:", "N/A" if rr is None else "%.2f" % rr)
        print("Winners:", plus, " Avg. Win:", "N/A" if avg_win is None else "%.2f" % avg_win + "%")
        print("Losers:", minus, " Avg. Loss:", "N/A" if avg_loss is None else "%.2f" % avg_loss + "%")
        print("")

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


def gen_add_plot(chart_data, entries, exits):
    df = chart_data.copy()

    n = len(df)

    data = {'Time': df['Time'].tolist(),
            'Price': [float('nan')] * n}
    df_markers = pd.DataFrame(data)
    df_markers = df_markers.set_index(pd.Index(df_markers.Time))
    df = df.set_index(pd.Index(df.Time))

    n1 = len(entries)
    n2 = len(exits)

    adp = []

    for i in range(0, n1):
        # marker for buy
        df_markers.loc[df.loc[entries[i][0]]['Time'], 'Price'] = entries[i][1]
        adp.append(mpf.make_addplot(df_markers['Price'].tolist(), scatter=True, markersize=120, marker=r'$\Rightarrow$', color='green'))
        df_markers.loc[df.loc[entries[i][0]]['Time'], 'Price'] = float('nan')

    for i in range(0, n2):

        # marker for sell
        df_markers.loc[df.loc[exits[i][0]]['Time'], 'Price'] = exits[i][1]
        adp.append(mpf.make_addplot(df_markers['Price'].tolist(), scatter=True, markersize=120, marker=r'$\Rightarrow$', color='red'))
        df_markers.loc[df.loc[exits[i][0]]['Time'], 'Price'] = float('nan')

    return adp


def show_1min_chart(df, symbol, date, info, entries, exits, rewards_b, rewards_i, save_to_dir=None, filename=None):
    df = df.set_index(pd.Index(df.Time))

    #############################################
    # Generate Add-Plots

    mav_list = []
    adp = gen_add_plot(df, entries, exits)

    # df['rewards_b'] = rewards_b
    # adp.append(mpf.make_addplot(df['rewards_b'], color='green'))

    # df['rewards_i'] = rewards_i
    # adp.append(mpf.make_addplot(df['rewards_i'], color='yellow'))

    ##################################
    # Plot charts

    title = info + symbol + " " + date

    if save_to_dir is None:  # Display chart
        if len(adp) > 0:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume', mav=mav_list,
                     volume=True, figscale=1, figratio=[16, 9], addplot=adp, title=title)
        else:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume', mav=mav_list,
                     volume=True, figscale=1, figratio=[16, 9], title=title)
    else:  # Save Chart in file
        make_dir(save_to_dir)
        if filename is None:
            path = save_to_dir + "\\" + symbol + '_' + date + ".png"
        else:
            path = save_to_dir + "\\" + filename + ".png"

        if len(adp) > 0:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume', mav=mav_list,
                     savefig=path, volume=True, figscale=2, figratio=[16, 9], addplot=adp, title=title)
        else:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume', mav=mav_list,
                     savefig=path, volume=True, figscale=2, figratio=[16, 9], title=title)


def get_time_index(df, date, h, m, s):
    idx = None

    xdate = pd.to_datetime(date, format="%Y-%m-%d")

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


def get_chart_data_prepared_for_ai(symbol, date):
    date = str(date).replace("-", "")
    df = get_intraday_chart_for(symbol, date)

    if df is not None:
        df.Time = pd.to_datetime(df.Time, format="%Y-%m-%d  %H:%M:%S")
        xdate = pd.to_datetime(date, format="%Y-%m-%d")

        idx = get_time_index(df, date, 15, 59, 0)

        if idx is not None:
            df = df[:idx+1]

        idx = get_time_index(df, date, 9, 30, 0)
        if idx is not None:
            df = df[idx:]

        n = len(df)
        xidx = None
        for i in range(0, n):
            if df.iloc[i]["Time"].date() != xdate:
                xidx = i
        if xidx is not None:
            df = df[xidx+1:]

        df.reset_index(drop=True, inplace=True)

    return df


def get_list_of_files(dir_path):
    symbols = []

    if os.path.isdir(dir_path):
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                symbols.append(name.replace(".csv", ""))

    return symbols


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


def get_daily_chart_for(symbol):
    path = __daily_charts_dir + "\\" + symbol + ".csv"

    df = None

    if os.path.isfile(path):
        df = pd.read_csv(path)
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


def get_intraday_chart_for(symbol, date):
    date = date.replace("-", "")

    path = __intraday_charts_dir + "\\" + symbol + "\\" + symbol + "_" + date + ".csv"

    df = None
    if os.path.isfile(path):
        df = pd.read_csv(path)
    else:
        print(colored('Warning! Intraday chart for ' + symbol + ' not available', color='yellow'))

    return df


def show_chart(symbol):
    df = get_daily_chart_for(symbol)

    df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index(pd.Index(df.Time))

    mpf.plot(df, type='candle', style="charles", volume=True)
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


def normalize(x):
    mn = min(x)
    mx = max(x)
    range_ = mx - mn
    # print(mn, mx)
    x = x - mn
    if range_ > 0:
        x = x / range_
    return x
