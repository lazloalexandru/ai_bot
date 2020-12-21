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
    zzz = np.fromfile('data\\datasets\\extended_dataset_0', dtype='float')

    DATA_SIZE = 2561
    n = len(zzz)
    rows = int(n / DATA_SIZE)

    print("Num Samples:", rows)

    zzz = zzz.reshape(rows, DATA_SIZE)

    print(zzz.shape)
    print(zzz[0], zzz[0][-1])
    print(zzz[1], zzz[1][-1])

    xidx = 300
    state = zzz[xidx][:-1]

    print("state: ", state.shape)
    symbol = "AAL"
    date = "2020-04-13"
    df = cu.get_intraday_chart_for(symbol, date)
    t = df.Time.to_list()
    chart.save_state_chart(state, t, "AAL", date, xidx, chart.EXTENDED_CHART_LENGTH)

    cu.show_daily_chart('CGC')
    cu.show_intraday_chart('CGC', '2018-08-16')


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
                    data_size = len(labeled_data)
                    if data_size > 0:
                        if data_size == chart.EXT_DATA_SIZE:
                            labeled_trades.append(labeled_data)
                        else:
                            print(colored("Algorithm ERROR!", color="red"))
                            return

                    if len(labeled_trades) >= num_samples_per_dataset:
                        print("")
                        xxx = np.array(labeled_trades)
                        print("Labeled Dataset Size1:", len(labeled_trades), "   ", xxx.shape)
                        xxx.tofile(dataset_path + str(dataset_id))
                        dataset_id += 1
                        labeled_trades = []
                        print("Labeled Dataset Size11:", len(labeled_trades))

            first_chart_idx += charts_per_batch

        if len(labeled_trades) > 0:
            xxx = np.array(labeled_trades)
            print("Labeled Dataset Size2:", len(labeled_trades), "   ", xxx.shape)
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
                data_size = len(labeled_data)
                if data_size > 0:
                    if data_size == chart.EXT_DATA_SIZE:
                        labeled_trades.append(labeled_data)
                    else:
                        print(colored("Algorithm ERROR!", color="red"))
                        return

                if len(labeled_trades) >= num_samples_per_dataset:
                    print("")
                    xxx = np.array(labeled_trades)
                    print("Labeled Dataset Size1:", len(labeled_trades), "   ", xxx.shape)
                    xxx.tofile(dataset_path + str(dataset_id))
                    dataset_id += 1
                    labeled_trades = []
                    print("Labeled Dataset Size11:", len(labeled_trades))

        if len(labeled_trades) > 0:
            xxx = np.array(labeled_trades)
            print("Labeled Dataset Size2:", len(labeled_trades), "   ", xxx.shape)
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
        if os.path.isdir(temp_dir_path):
            try:
                os.rename(temp_dir_path, result_images_dir_path)
            except OSError:
                print("Renaming", temp_dir_path, "to", result_images_dir_path, " failed!")


def _gen_dataset_from_chart(c, params):
    symbol = c['symbol']
    date = str(c['date']).replace("-", "")

    dataset = []

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

        df_daily = cu.get_daily_chart_for(symbol)
        df_history, date_index = cu.get_period_before(df_daily, date, chart.DAILY_CHART_LENGTH, symbol + "_" + str(date))
        df = cu.get_chart_data_prepared_for_ai(symbol, date, params)

        open_index = cu.get_time_index(df, date, params['__chart_begin_hh'], params['__chart_begin_mm'], 0)

        if df is not None and open_index is not None and df_history is not None and date_index is not None:
            params['symbol'] = symbol
            params['date'] = date
            params['date_index'] = date_index
            entries, dataset = _gen_labeled_data_from_chart(df_history, df, params)

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


def calc_range(df_chart):
    min_price = min(df_chart.Low.to_list())
    max_price = max(df_chart.High.to_list())
    if min_price <= 0:
        min_price = 0.1

    return 100*(max_price / min_price - 1)


def _gen_labeled_data_from_chart(df_history, df_chart, params):
    entries = []
    dataset = []

    date = params['date']
    open_index = cu.get_time_index(df_chart, date, params['__chart_begin_hh'], params['__chart_begin_mm'], 0)
    close_index = cu.get_time_index(df_chart, date, params['__chart_end_hh'], params['__chart_end_mm'], 0)
    trading_start_idx = cu.get_time_index(df_chart, date, params['trading_begin_hh'], params['trading_begin_mm'], 0)

    if trading_start_idx is None:
        trading_start_idx = df_chart.index[0]

    params['stop_sell'] = - calc_range(df_chart) / params['stop_sell_factor']
    params['stop_buy'] = calc_range(df_chart) / params['stop_buy_factor']

    min_price = df_chart['Low'][open_index]
    max_price = df_chart['High'][open_index]
    vol = cu.get_premarket_volume_for(params)

    params['Open'] = df_chart['Open'][open_index]

    i = open_index
    while i <= close_index:
        if df_chart['Low'][i] < min_price:
            min_price = df_chart['Low'][i]
        if df_chart['High'][i] > max_price:
            max_price = df_chart['High'][i]

        vol += df_chart['Volume'][i]

        if i >= trading_start_idx:
            ####################### Set Up Values For Dynamic Daily Candle #####################
            params['Close'] = df_chart['Close'][i]
            params['High'] = max_price
            params['Low'] = min_price
            params['Volume'] = vol
            chart.update_candle(df_history, params)
            ###################################################################################

            buy_price = df_chart['Close'][i]

            params['stop_SELL_price'] = buy_price * (1 + params['stop_sell'] / 100)
            params['stop_BUY_price'] = buy_price * (1 + params['stop_buy'] / 100)

            state, label = _gen_labeled_data_for_entry(df_history, df_chart, i, open_index, params)

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


def _gen_labeled_data_for_entry(df_history, df_chart, entry_index, open_index, params):
    gain = _max_win_for_entry(df_chart, entry_index, open_index, params)

    '''
    if gain < 2:
        gain = _max_loss_for_entry(df_chart, entry_index, open_index, params)
    '''

    intra_day_state, label = _gen_labeled_data(df_history, df_chart, entry_index, open_index, gain, params)

    state = intra_day_state  # + daily state

    return state, label


def _gen_labeled_data(df_history, df, entry_idx, open_idx, gain, params):
    state = chart.create_state_vector(df_history, df, entry_idx, open_idx)

    label = 0

    target = abs(2 * params['stop_sell'])
    if gain < target:
        label = 0
    elif target <= gain:
        label = 1

    return state, label


def get_marker(label):
    m = '$' + str(label) + '$'

    if label == 0:
        c = 'red'
    elif label == 1:
        c = 'green'

    return m, c


def main():
    params = get_default_params()

    # params['filter_sym'] = 'DENN'
    # params['filter_date'] = '2020-03-23'
    # test_training_data()

    generate_datasets_mp(params)


def get_default_params():
    params = {
        '__chart_begin_hh': 9,
        '__chart_begin_mm': 30,
        '__chart_end_hh': 15,
        '__chart_end_mm': 59,

        'trading_begin_hh': 9,
        'trading_begin_mm': 40,

        'stop_buy_factor': 6,
        'stop_sell_factor': 6,


        'no_charts': True,

        'chart_list_file': "data\\training_charts.csv",
        'dataset_name': "training_data",
        'charts_per_batch': 200,
        'num_samples_per_dataset': 1000000,

        'num_cores': 16
    }
    return params


if __name__ == "__main__":
    main()
