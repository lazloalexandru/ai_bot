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
                     savefig=path, volume=True, figscale=2, figratio=[16, 9], addplot=adp, title=title, tight_layout=True)
        else:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume',
                     savefig=path, volume=True, figscale=2, figratio=[16, 9], title=title, tight_layout=True)


def generate_labeled_data_mp(params):
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

            first_chart_idx += charts_per_batch

            for res in mp_results:
                dataset = res.get(timeout=1)
                for labeled_data in dataset:
                    data_size = len(labeled_data)
                    if data_size > 0:
                        if data_size == chart.EXT_DATA_SIZE:
                            labeled_trades.append(labeled_data)
                        else:
                            print(colored("Algorithm ERROR! " + str(data_size), color="red"))

                    if len(labeled_trades) >= num_samples_per_dataset:
                        print("")
                        xxx = np.array(labeled_trades)
                        print("Labeled Dataset Size1:", len(labeled_trades), "   ", xxx.shape)
                        np.save(dataset_path + str(dataset_id), xxx)
                        dataset_id += 1
                        labeled_trades = []
                        print("Labeled Dataset Size11:", len(labeled_trades))

        if len(labeled_trades) > 0:
            xxx = np.array(labeled_trades)
            print("Labeled Dataset Size2:", len(labeled_trades), "   ", xxx.shape)
            np.save(dataset_path + str(dataset_id), xxx)
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
                        print(colored("Algorithm ERROR! " + str(data_size), color="red"))

                if len(labeled_trades) >= num_samples_per_dataset:
                    print("")
                    xxx = np.array(labeled_trades)
                    print("Labeled Dataset Size1:", len(labeled_trades), "   ", xxx.shape)
                    np.save(dataset_path + str(dataset_id), xxx)
                    dataset_id += 1
                    labeled_trades = []
                    print("Labeled Dataset Size11:", len(labeled_trades))

        if len(labeled_trades) > 0:
            xxx = np.array(labeled_trades)
            print("Labeled Dataset Size2:", len(labeled_trades), "   ", xxx.shape)
            print(xxx[0].shape)
            print(dataset_path + str(dataset_id))
            np.save(dataset_path + str(dataset_id), xxx)
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

        labels, label_type = cu.get_automatic_labels(symbol, date)

        if df is not None and open_index is not None and labels is not None:  # and label_type == 1:
            params['symbol'] = symbol
            params['date'] = date
            params['date_index'] = date_index
            entries, dataset = _gen_labeled_data_from_chart(df_history, df, labels, params)

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


def _gen_labeled_data_from_chart(df_history, df_chart, labels, params):
    entries = []
    dataset = []

    date = params['date']
    open_index = cu.get_time_index(df_chart, date, params['__chart_begin_hh'], params['__chart_begin_mm'], 0)
    close_index = cu.get_time_index(df_chart, date, params['__chart_end_hh'], params['__chart_end_mm'], 0)
    trading_start_idx = cu.get_time_index(df_chart, date, params['trading_begin_hh'], params['trading_begin_mm'], 0)

    if trading_start_idx is None:
        trading_start_idx = df_chart.index[0]

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

            state = chart.create_state_vector(df_history, df_chart, i, open_index)
            label = labels[i]

            labeled_data = np.concatenate((state, [label]))
            dataset.append(labeled_data)

            entries.append([df_chart['Time'][i], df_chart['High'][i], label])

        i = i + 1

    return entries, dataset


def main():
    params = get_default_params()

    # params['filter_sym'] = 'DENN'
    # params['filter_date'] = '2020-03-23'
    # test_training_data()

    generate_labeled_data_mp(params)


def get_default_params():
    params = {
        '__chart_begin_hh': 9,
        '__chart_begin_mm': 30,
        '__chart_end_hh': 15,
        '__chart_end_mm': 59,

        'trading_begin_hh': 9,
        'trading_begin_mm': 40,

        'no_charts': True,

        'chart_list_file': "data\\test_charts.csv",
        'dataset_name': "test_data",
        'charts_per_batch': 300,
        'num_samples_per_dataset': 1000000,

        'num_cores': 16
    }
    return params


if __name__ == "__main__":
    main()
