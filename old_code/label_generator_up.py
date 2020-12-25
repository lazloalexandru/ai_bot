import multiprocessing as mp
from termcolor import colored


import chart
import common as cu
import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np

___temp_dir_name = "__temp__"


def cmean(x, half_period):
    n = len(x)
    avg = np.zeros(n)
    for i in range(n):
        s = 0
        m = 0
        for j in range(i-half_period, i+half_period+1):
            if 0 <= j < n:
                s += x[j]
                m += 1
        avg[i] = s / m

    return avg


def _gen_add_plot(chart_data, entries):
    df = chart_data.copy()

    data = {
        'Time': df['Time'].tolist(),
        'Price': [float('nan')] * len(df)
    }

    df_markers = pd.DataFrame(data)
    df_markers = df_markers.set_index(pd.Index(df_markers.Time))
    df = df.set_index(pd.Index(df.Time))

    n = len(entries)

    adp = []

    for i in range(0, n):
        m, c = chart.get_marker(entries[i][2])

        df_markers.loc[df.loc[entries[i][0]]['Time'], 'Price'] = entries[i][1] + 0.05
        adp.append(mpf.make_addplot(df_markers['Price'].tolist(), scatter=True, markersize=20, marker=m, color=c))
        df_markers.loc[df.loc[entries[i][0]]['Time'], 'Price'] = float('nan')

    return adp


def _show_chart(chart_data, symbol, date, info, entries, save_to_dir=""):
    df = chart_data.copy()

    df = df.set_index(pd.Index(df.Time))

    adp = _gen_add_plot(chart_data, entries)
    adp.append(mpf.make_addplot(df["avg_s"].tolist(), color='yellow'))
    adp.append(mpf.make_addplot(df["avg_l"].tolist(), color='blue'))

    plt.rcParams['figure.dpi'] = 240
    plt.rcParams['figure.figsize'] = [1.0, 1.0]

    title = info + symbol + " " + date

    if save_to_dir == "":  # Display chart
        if len(adp) > 0:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume',
                     volume=True, figscale=1, figratio=[16, 9], addplot=adp, title=title, tight_layout=True)
        else:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume',
                     volume=True, figscale=1, figratio=[16, 9], title=title, tight_layout=True)
    else:  # Save Chart in file
        cu.make_dir(save_to_dir)
        path = save_to_dir + "\\" + symbol + '_' + date + ".png"

        if len(adp) > 0:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume',
                     savefig=path, volume=True, figscale=2, figratio=[16, 9], addplot=adp, title=title, tight_layout=True)
        else:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume',
                     savefig=path, volume=True, figscale=2, figratio=[16, 9], title=title, tight_layout=True)


def generate_labels_mp(params):
    filter_mode = 'filter_sym' in params.keys() and 'filter_date' in params.keys()
    no_charts = 'no_charts' in params.keys() and params['no_charts']

    ################################################

    chart_list_file_path = params['chart_list_file']
    if not os.path.isfile(chart_list_file_path):
        print(colored('Warning! Cannot find chart list file >>> ' + chart_list_file_path, color='yellow'))
        return

    df_charts = pd.read_csv(chart_list_file_path)
    num_charts = len(df_charts)

    print("Number of charts:", num_charts)

    ################################################
    # Init directory structure

    temp_dir_path = cu.__labels_dir + "\\" + ___temp_dir_name
    cu.erase_dir_if_exists(temp_dir_path)

    if not filter_mode:
        if not no_charts:
            print("Saving charts at " + temp_dir_path)

    #################################################
    # Processing ...
    cpu_count = params['num_cores']

    if cpu_count > 1 and not filter_mode:
        print("Num CPUs: %s\n" % cpu_count)

        pool = mp.Pool(cpu_count)
        for i in range(num_charts):
            pool.apply_async(_gen_labels_for_chart, args=(df_charts.loc[i], params))

        pool.close()
        pool.join()
    else:
        print("Single CPU execution", end="")

        if filter_mode:
            print(" [ Filter_Symbol:", params['filter_sym'], "  Filter_Date:", params['filter_date'], "]")
        else:
            print("")

        for index, row in df_charts.iterrows():
            _gen_labels_for_chart(row, params)

    ###############################################################
    # save images

    result_images_dir_path = cu.__labels_dir + "\\" + params['label_name']
    cu.erase_dir_if_exists(result_images_dir_path)

    if not no_charts:
        if os.path.isdir(temp_dir_path):
            try:
                os.rename(temp_dir_path, result_images_dir_path)
            except OSError:
                print("Renaming", temp_dir_path, "to", result_images_dir_path, " failed!")


def _gen_labels_for_chart(chart_info, params):
    symbol = chart_info['symbol']
    date = str(chart_info['date']).replace("-", "")

    filter_mode_on = 'filter_sym' in params.keys() and 'filter_date' in params.keys()
    no_charts = 'no_charts' in params.keys() and params['no_charts']

    search_needed = False

    if not filter_mode_on:
        search_needed = True
    elif filter_mode_on and symbol == params['filter_sym']:
        if date == str(params['filter_date']).replace("-", ""):
            search_needed = True

    if search_needed:
        print(symbol + ' ' + date)
        df = cu.get_chart_data_prepared_for_ai(symbol, date, params)

        open_index = cu.get_time_index(df, date, params['__chart_begin_hh'], params['__chart_begin_mm'], 0)

        if df is not None and open_index is not None:
            xxx = df["Close"].tolist()
            df["avg_s"] = cmean(xxx, 5)
            df["avg_l"] = cmean(xxx, 10)

            params['symbol'] = symbol
            params['date'] = date
            entries = _gen_labels_from_chart_data(df, params)

            if len(entries) > 0 or filter_mode_on:
                if not no_charts:
                    images_dir_path = cu.__labels_dir + "\\" + ___temp_dir_name
                    _show_chart(chart_data=df,
                                symbol=symbol,
                                date=date,
                                info="",
                                entries=entries,
                                save_to_dir="" if filter_mode_on else images_dir_path)


def _gen_labels_from_chart_data(df_chart, params):
    entries = []

    labels = np.zeros(len(df_chart))

    date = params['date']
    symbol = params['symbol']
    open_index = cu.get_time_index(df_chart, date, params['__chart_begin_hh'], params['__chart_begin_mm'], 0)
    close_index = cu.get_time_index(df_chart, date, params['__chart_end_hh'], params['__chart_end_mm'], 0)
    trading_start_idx = cu.get_time_index(df_chart, date, params['trading_begin_hh'], params['trading_begin_mm'], 0)

    if trading_start_idx is None:
        trading_start_idx = df_chart.index[0]

    min_price = df_chart['Low'][open_index]
    max_price = df_chart['High'][open_index]

    i = open_index
    while i <= close_index:
        if df_chart['Low'][i] < min_price:
            min_price = df_chart['Low'][i]
        if df_chart['High'][i] > max_price:
            max_price = df_chart['High'][i]

        params['stop'] = cu.calc_range(min_price, max_price) / params['stop_factor']

        if i >= trading_start_idx:

            buy_price = df_chart['Close'][i]

            params['stop_SELL_price'] = buy_price * (1 - params['stop'] / 100)
            params['stop_BUY_price'] = buy_price * (1 + params['stop'] / 100)

            label = _gen_label_for_entry(df_chart, i, params)

            labels[i] = label
            entries.append([df_chart['Time'][i], df_chart['High'][i], label])

        i = i + 1

    np.save(cu.get_generated_label_path_for(symbol, date), labels)

    return entries


def _max_win_for_entry(df_chart, entry_index, params):
    sold = False
    sell_price = None

    entry_price = df_chart['Close'][entry_index]
    chart_end_idx = df_chart.index[-1]

    max_price = -1
    j = entry_index + 1

    while j < chart_end_idx and not sold:
        if df_chart['Low'][j] < params['stop_SELL_price']:
            sold = True
            sell_price = params['stop_SELL_price']
        elif df_chart['High'][j] > max_price:
            max_price = df_chart['High'][j]
        j = j + 1

    if sold:
        if max_price > 0:
            sell_price = max_price
    else:
        if j == chart_end_idx:
            if max_price > 0:
                sell_price = max_price
            else:
                sell_price = df_chart.loc[chart_end_idx]['High']
        elif j > chart_end_idx:
            sell_price = df_chart.loc[chart_end_idx]['High']

    return int(100 * (sell_price - entry_price) / entry_price)


def _max_loss_for_entry(df_chart, entry_index, params):
    sold = False
    sell_price = None

    chart_end_idx = df_chart.index[-1]

    VERY_BIG_PRICE = 1000000
    min_price = VERY_BIG_PRICE

    j = entry_index + 1

    while j < chart_end_idx and not sold:
        if df_chart['High'][j] > params['stop_BUY_price']:
            sold = True
            sell_price = params['stop_BUY_price']

        elif df_chart['Low'][j] < min_price:
            min_price = df_chart['Low'][j]
        j = j + 1

    if sold:
        if min_price < VERY_BIG_PRICE:
            sell_price = min_price
    else:
        if j == chart_end_idx:
            if min_price < VERY_BIG_PRICE:
                sell_price = min_price
            else:
                sell_price = df_chart.loc[chart_end_idx]['Low']
        elif j > chart_end_idx:
            sell_price = df_chart.loc[chart_end_idx]['Low']

    entry_price = df_chart['Close'][entry_index]

    return int(100 * (sell_price - entry_price) / entry_price)


def _gen_label_for_entry(df_chart, entry_index, params):
    gain = _max_win_for_entry(df_chart, entry_index, params)

    label = 0 if gain < (params['R/R'] * params['stop']) else 1

    return label


def main():
    params = get_default_params()

    # params['filter_sym'] = 'DENN'
    # params['filter_date'] = '2020-03-23'
    # test_training_data()

    generate_labels_mp(params)


def get_default_params():
    params = {
        '__chart_begin_hh': 9,
        '__chart_begin_mm': 30,
        '__chart_end_hh': 15,
        '__chart_end_mm': 59,

        'trading_begin_hh': 9,
        'trading_begin_mm': 40,

        'R/R': 1,
        'stop_factor': 6,

        'no_charts': False,

        'chart_list_file': "data\\all_tradeable_charts.csv",
        'label_name': "xxxx_auto_labeled",
        'num_cores': 16
    }
    return params


if __name__ == "__main__":
    main()
