import multiprocessing as mp
from termcolor import colored

import chart
import common as cu
import os
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import numpy as np
import gc

___temp_dir_name = "__temp__"


def test_training_data():
    zzz = np.fromfile('data\\datasets\\dataset_2', dtype='float')

    n = len(zzz)
    rows = int(n / 1951)

    print("Num Samples:", rows)

    zzz = zzz.reshape(rows, 1951)

    print(zzz.shape)
    print(zzz[0], zzz[0][-1])
    print(zzz[1], zzz[1][-1])

    state = zzz[4100][:-1]

    print("state: ", state.shape)
    symbol = "AAL"
    date = "2020-04-13"
    df = cu.get_chart_data_prepared_for_ai(symbol, date, get_default_params())
    t = df.Time.to_list()
    chart.save_state_chart(state, t, "----", date, 1)


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
        m, c = get_marker(entries[i][2])

        df_markers.loc[df.loc[entries[i][0]]['Time'], 'Price'] = entries[i][1] + 0.05
        adp.append(mpf.make_addplot(df_markers['Price'].tolist(), scatter=True, markersize=20, marker=m, color=c))
        df_markers.loc[df.loc[entries[i][0]]['Time'], 'Price'] = float('nan')

    return adp


def _show_chart(chart_data, symbol, date, info, entries, params, save_to_dir=""):
    df = chart_data.copy()

    df = df.set_index(pd.Index(df.Time))

    adp = _gen_add_plot(chart_data, entries)

    plt.rcParams['figure.dpi'] = 240
    plt.rcParams['figure.figsize'] = [1.0, 1.0]

    title = info + symbol + " " + date

    if save_to_dir == "":  # Display chart
        if len(adp) > 0:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume',
                     volume=True, figscale=1, figratio=[16, 9], addplot=adp, title=title)
        else:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume',
                     volume=True, figscale=1, figratio=[16, 9], title=title)
    else:  # Save Chart in file
        cu.make_dir(save_to_dir)
        path = save_to_dir + "\\" + symbol + '_' + date + ".png"

        if len(adp) > 0:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume',
                     savefig=path, volume=True, figscale=2, figratio=[16, 9], addplot=adp, title=title)
        else:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume',
                     savefig=path, volume=True, figscale=2, figratio=[16, 9], title=title)


def generate_datasets_mp(params):
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

    cu.make_dir(cu.__datasets_dir)

    temp_dir_path = cu.__datasets_dir + "\\" + ___temp_dir_name
    cu.erase_dir_if_exists(temp_dir_path)

    if not filter_mode:
        if not no_charts:
            print("Saving charts at " + temp_dir_path)

        #################################################
        # Processing ...

        version = 'na'

        labeled_trades = []
        dataset_id = 0
        num_samples_per_dataset = params['num_samples_per_dataset']
        dataset_path = cu.__datasets_dir + "\\" + params['dataset_name'] + "_"
        print("Saving labeled trades to file (%s samples): %sxxx" % (num_samples_per_dataset, dataset_path))

        df_fund = cu.get_fundamentals()

        cpu_count = params['num_cores']

        if cpu_count > 1 and not filter_mode:
            print("Num CPUs: %s\n" % cpu_count)


            first_chart_idx = 0
            charts_per_batch = params['charts_per_batch']
            while first_chart_idx < num_charts:

                last_chart_idx = first_chart_idx + charts_per_batch
                if last_chart_idx > num_charts:
                    last_chart_idx = num_charts

                pool = mp.Pool(cpu_count)
                mp_results = [
                    pool.apply_async(_gen_dataset_from_chart,
                                     args=(df_charts.loc[i], params)
                                     ) for i in range(first_chart_idx, last_chart_idx)
                ]

                pool.close()
                pool.join()

                for res in mp_results:
                    dataset = res.get(timeout=1)
                    for labeled_data in dataset:
                        labeled_trades.append(labeled_data)

                        if len(labeled_trades) >= num_samples_per_dataset:
                            xxx = np.array(labeled_trades)
                            print("Labeled Dataset Size:", len(labeled_trades), "   ", xxx.shape)
                            xxx.tofile(dataset_path + str(dataset_id))
                            dataset_id += 1
                            labeled_trades = []

                first_chart_idx += charts_per_batch

            if len(labeled_trades) > 0:
                xxx = np.array(labeled_trades)
                print("Labeled Dataset Size:", len(labeled_trades), "   ", xxx.shape)
                xxx.tofile(dataset_path + str(dataset_id))
                dataset_id += 1
        else:
            print("Single CPU execution", end="")

            if filter_mode:
                print(" [ Filter_Symbol:", params['filter_sym'], "  Filter_Date:", params['filter_date'], "]")
            else:
                print("")

            for index, row in df_charts.iterrows():
                dataset = _gen_dataset_from_chart(row, params)
                for labeled_data in dataset:
                    labeled_trades.append(labeled_data)

                    if len(labeled_trades) >= num_samples_per_dataset:
                        xxx = np.array(labeled_trades)
                        print("Labeled Dataset Size:", len(labeled_trades), "   ", xxx.shape)
                        xxx.tofile(dataset_path + str(dataset_id))
                        dataset_id += 1
                        labeled_trades = []

            if len(labeled_trades) > 0:
                xxx = np.array(labeled_trades)
                print("Labeled Dataset Size:", len(labeled_trades), "   ", xxx.shape)
                print(xxx[0].shape)
                print(dataset_path + str(dataset_id))
                xxx.tofile(dataset_path + str(dataset_id))
                dataset_id += 1

        params['version'] = version

        ###############################################################
        # save results

        result_images_dir_path = cu.__datasets_dir + "\\" + params['dataset_name']

        cu.erase_dir_if_exists(result_images_dir_path)

        if not no_charts:
            try:
                os.rename(temp_dir_path, result_images_dir_path)
            except OSError:
                print("Renaming", temp_dir_path, "to", result_images_dir_path, " failed!")


def _gen_dataset_from_chart(c, params):
    dataset = []

    filter_mode_on = 'filter_sym' in params.keys() and 'filter_date' in params.keys()
    no_charts = 'no_charts' in params.keys() and params['no_charts']

    search_needed = False

    if not filter_mode_on:
        search_needed = True
    elif filter_mode_on and c['symbol'] == params['filter_sym'] and str(c['date']) == str(params['filter_date']):
        search_needed = True

    if search_needed:
        symbol = c['symbol']
        date = str(c['date'])

        print(symbol + ' ' + date)

        df = cu.get_chart_data_prepared_for_ai(symbol, date, params)

        open_index = cu.get_time_index(df, date, params['__chart_begin_hh'], params['__chart_begin_mm'], 0)

        if df is not None and open_index is not None:
            params['symbol'] = symbol
            params['date'] = date
            entries, dataset = _gen_labeled_data_from_chart(df, params)

            info_text = ""

            if len(entries) > 0 or filter_mode_on:
                if not no_charts:
                    images_dir_path = cu.__datasets_dir + "\\" + ___temp_dir_name
                    _show_chart(chart_data=df,
                                symbol=symbol,
                                date=date,
                                info=info_text,
                                entries=entries,
                                params=params,
                                save_to_dir="" if filter_mode_on else images_dir_path)

    return dataset


def _gen_labeled_data_from_chart(df_chart, params):
    entries = []
    dataset = []

    date = params['date']
    open_index = cu.get_time_index(df_chart, date, params['__chart_begin_hh'], params['__chart_begin_mm'], 0)
    close_index = cu.get_time_index(df_chart, date, params['__chart_end_hh'], params['__chart_end_mm'], 0)
    trading_start_idx = cu.get_time_index(df_chart, date, params['trading_begin_hh'], params['trading_begin_mm'], 0)

    if trading_start_idx is None:
        trading_start_idx = df_chart.index[0]

    i = trading_start_idx
    while i <= close_index:
        buy_price = df_chart['Close'][i]

        params['stop_SELL_price'] = buy_price * (1 + params['stop_sell'] / 100)
        params['stop_BUY_price'] = buy_price * (1 + params['stop_buy'] / 100)

        state, label = _gen_labeled_data_for_entry(df_chart, i, open_index, params)

        labeled_data = np.concatenate((state, [label]))
        dataset.append(labeled_data)

        entries.append([df_chart['Time'][i], df_chart['High'][i], label])

        i = i + 1

    return entries, dataset


def _max_win_for_entry(df_chart, entry_index, open_index, params):
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
            max_idx = j

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


def _max_loss_for_entry(df_chart, entry_index, open_index, params):
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


def _gen_labeled_data_for_entry(df_chart, entry_index, open_index, params):
    gain = _max_win_for_entry(df_chart, entry_index, open_index, params)

    if gain < 2:
        gain = _max_loss_for_entry(df_chart, entry_index, open_index, params)

    state, label = _gen_labeled_data(df_chart, entry_index, open_index, gain)

    return state, label


def _gen_labeled_data(df, entry_idx, open_idx, gain):
    state = chart.create_padded_state_vector(df, entry_idx, open_idx)

    label = 0

    if gain < -10:
        label = 0
    elif -10 <= gain < -5:
        label = 1
    elif -5 <= gain < -2:
        label = 2
    elif -2 <= gain < 2:
        label = 3
    elif 2 <= gain < 5:
        label = 4
    elif 5 <= gain < 10:
        label = 5
    elif 10 <= gain:
        label = 6

    return state, label


def get_marker(label):
    m = '$' + str(label) + '$'

    if label == 0:
        c = 'darkred'
    elif label == 1:
        c = 'red'
    elif label == 2:
        c = 'orange'
    elif label == 3:
        c = 'yellow'
    elif label == 4:
        c = 'limegreen'
    elif label == 5:
        c = 'seagreen'
    elif label == 6:
        c = 'darkgreen'

    return m, c


def main():
    params = get_default_params()

    # params['filter_sym'] = 'ADAP'
    # params['filter_date'] = '20200114'

    generate_datasets_mp(params)
    # test_training_data()


def get_default_params():
    params = {
        '__chart_begin_hh': 9,
        '__chart_begin_mm': 30,
        '__chart_end_hh': 15,
        '__chart_end_mm': 59,

        'trading_begin_hh': 9,
        'trading_begin_mm': 40,

        'stop_buy': 2,
        'target_short': -10,

        'stop_sell': -5,
        'target_long': 10,


        'no_charts': True,

        'chart_list_file': "data\\test_charts.csv",
        'dataset_name': "test_dataset",
        'charts_per_batch': 500,
        'num_samples_per_dataset': 2500000,

        'num_cores': 6
    }
    return params


if __name__ == "__main__":
    main()
