from termcolor import colored

import chart
import common as cu
import os
import pandas as pd
import mplfinance as mpf
import sim_account as sim
import matplotlib.pyplot as plt
from model import Net
import numpy as np
import torch


___temp_dir_name = "__temp__"


def _gen_add_plot(chart_data, entries, exits):
    df = chart_data.copy()
    n = len(df)

    data = {
        'Time': df['Time'].tolist(),
        'Price': [float('nan')] * n
    }
    df_markers = pd.DataFrame(data)
    df_markers = df_markers.set_index(pd.Index(df_markers.Time))
    df = df.set_index(pd.Index(df.Time))

    n1 = len(entries)
    n2 = len(exits)

    adp = []

    if n1 == n2:
        if n1 > 0:
            pivots = entries[0][2]
            n = len(pivots)
            for j in range(0, n):
                df_markers.loc[df.loc[pivots[j][2]]['Time'], 'Price'] = pivots[j][0]
                adp.append(mpf.make_addplot(df_markers['Price'].tolist(), scatter=True, markersize=100, marker=r'$*$', color='red' if pivots[j][1] else 'black'))
                df_markers.loc[df.loc[pivots[j][2]]['Time'], 'Price'] = float('nan')

        for i in range(0, n1):
            # marker for buy
            df_markers.loc[df.loc[entries[i][0]]['Time'], 'Price'] = entries[i][1]
            adp.append(mpf.make_addplot(df_markers['Price'].tolist(), scatter=True, markersize=120, marker=r'$\Rightarrow$', color='green'))
            df_markers.loc[df.loc[entries[i][0]]['Time'], 'Price'] = float('nan')

            # marker for sell
            df_markers.loc[df.loc[exits[i][0]]['Time'], 'Price'] = exits[i][1]
            adp.append(mpf.make_addplot(df_markers['Price'].tolist(), scatter=True, markersize=120, marker=r'$\Rightarrow$', color='red'))
            df_markers.loc[df.loc[exits[i][0]]['Time'], 'Price'] = float('nan')
    else:
        print(colored("Number of entries: " + str(n1) + " Number of exits: " + str(n2) + "  => Algorithm ERROR!!!! ", color='red'))

    return adp


def _show_1min_chart(chart_data, symbol, date, info, entries, exits, params, save_to_dir=""):
    df = chart_data.copy()

    df = df.set_index(pd.Index(df.Time))

    idx_b = cu.get_time_index(df, date, params['trading_begin_hh'], params['trading_begin_mm'], 0)
    if idx_b is None:
        idx_b = df.index[0]

    adp = _gen_add_plot(chart_data, entries, exits)

    title = info + symbol + " " + date

    if save_to_dir == "":  # Display chart
        if len(adp) > 0:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume', volume=True,
                     figscale=1, figratio=[16, 9], addplot=adp, vlines=[df.loc[idx_b]['Time']], title=title)
        else:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume', volume=True,
                     figscale=1, figratio=[16, 9], vlines=[df.loc[idx_b]['Time']], title=title)
    else:  # Save Chart in file
        cu.make_dir(save_to_dir)
        path = save_to_dir + "\\" + symbol + '_' + date + ".png"
        # path = save_to_dir + "\\" + date + '_' + symbol + ".png"

        if len(adp) > 0:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume', savefig=path, volume=True,
                     figscale=2, figratio=[16, 9], addplot=adp, vlines=[df.loc[idx_b]['Time']], title=title)
        else:
            mpf.plot(df, type='candle', ylabel='Price', ylabel_lower='Volume', savefig=path, volume=True,
                     figscale=2, figratio=[16, 9], vlines=[df.loc[idx_b]['Time']], title=title)


def simulate_pattern(params):
    sim_params = {
        'account_value': 10000,
        'size_limit': 50000,
        'hard_stop': params['stop']
    }

    params['pattern'] = 'ai'
    params['version'] = 3

    pattern_file_path = get_pattern_file_path(params)
    stats = sim.simulate_account_performance_for(sim_params, pattern_file_path)

    sim_report_file_path = pattern_file_path.replace(".csv", "_sim_report.txt")
    cu.save_simulation_report(sim_params, stats, sim_report_file_path)


def search_patterns(params):
    params['pattern'] = 'ai'

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

    print(df_charts.head())

    num_test_charts = int(num_charts * (1 - params['split_train_test']))
    first_chart_idx = int(num_charts * params['split_train_test']) + 1
    last_chart_idx = num_charts

    print("Test Set Charts:", num_test_charts, "  (%s,%s)" % (first_chart_idx, last_chart_idx))

    ################################################
    # Init directory structure

    pattern_dir_path = "trades\\" + params['pattern']
    cu.make_dir(pattern_dir_path)

    temp_dir_path = "trades\\" + params['pattern'] + "\\" + ___temp_dir_name
    cu.erase_dir_if_exists(temp_dir_path)

    if not filter_mode:
        print("Saving charts at " + temp_dir_path)

    #################################################
    # Processing ...

    version = 'na'
    results = []

    df_fund = cu.get_fundamentals()

    if df_fund is None:
        return

    print("Single CPU execution", end="")

    if filter_mode:
        print(" [ Filter_Symbol:", params['filter_sym'], "  Filter_Date:", params['filter_date'], "]")
    else:
        print("")

    for i in range(first_chart_idx, last_chart_idx):
        df, ver = _search_patterns_in(df_charts.loc[i], params, df_fund)
        if len(df) > 0:
            results.append(df)
            version = ver[0]

    params['version'] = version

    ###############################################################
    # save results

    result_filename = get_pattern_file_path(params)
    result_images_dir_path = result_filename.replace(".csv", "")

    filter_mode_on = 'filter_sym' in params.keys() and 'filter_date' in params.keys()

    if len(results) > 0 and not filter_mode_on:
        df_result = pd.concat(results)
        df_result = df_result.sort_values(by=['entry_time'])
        df_result.reset_index(drop=True, inplace=True)

        print("\nSaving results to file: " + result_filename)
        df_result.to_csv(result_filename)

        cu.erase_dir_if_exists(result_images_dir_path)

        if not no_charts:
            try:
                os.rename(temp_dir_path, result_images_dir_path)
            except OSError:
                print("Renaming", temp_dir_path, "to", result_images_dir_path, " failed!")
    else:
        print("\nResult: 0 entries found!")
        cu.erase_dir_if_exists(temp_dir_path)


def _search_patterns_in(gapper, params, df_fund):
    version = []

    df_result = pd.DataFrame(columns=['sym', 'date', 'entry_time', 'entry_price', 'stop', 'sell_time', 'sell_price', 'exit_type'])

    filter_mode_on = 'filter_sym' in params.keys() and 'filter_date' in params.keys()
    no_charts = 'no_charts' in params.keys() and params['no_charts']

    search_needed = False

    if not filter_mode_on:
        search_needed = True
    elif filter_mode_on and gapper['sym'] == params['filter_sym'] and gapper['date'] == params['filter_date']:
        search_needed = True

    if search_needed:
        symbol = gapper['symbol']
        date = str(gapper['date'])

        params['symbol'] = symbol
        params['date'] = date

        print('\n' + symbol + ' ' + date)

        df = cu.get_chart_data_prepared_for_ai(symbol, date, params)
        open_index = cu.get_time_index(df, date, params['__chart_begin_hh'], params['__chart_begin_mm'], 0)

        if df is not None and open_index is not None:
            entries, exits = _find_trades(df, params, version)

            n1 = len(entries)
            n2 = len(exits)

            info_text = ""
            fundamentals = cu.get_fundamentals_for(df_fund, symbol)
            if fundamentals is not None:
                info_text = info_text + "     Shares: %.1fM            " % fundamentals['shares_outstanding']

            if len(entries) > 0 or filter_mode_on:
                if not no_charts:
                    images_dir_path = "trades\\" + params['pattern'] + "\\" + ___temp_dir_name
                    _show_1min_chart(chart_data=df,
                                     symbol=symbol,
                                     date=date,
                                     info=info_text,
                                     entries=entries,
                                     exits=exits,
                                     params=params,
                                     save_to_dir="" if filter_mode_on else images_dir_path)

            if n1 == n2:
                for j in range(0, n1):
                    data = {'sym': symbol,
                            'date': date,
                            'entry_time': entries[j][0],
                            'entry_price': entries[j][1],
                            'stop': entries[j][3],
                            'sell_time': exits[j][0],
                            'sell_price': exits[j][1],
                            'exit_type': exits[j][2]}
                    df_result = df_result.append(data, ignore_index=True)
            else:
                print(colored("Number of entries: " + str(n1) + " Number of exits: " + str(n2) + "  => Algorithm ERROR!!!! ", color='red'))

    return df_result, version


def gen_state(df, entry_idx, open_idx):
    state = chart.create_padded_state_vector(df, entry_idx, open_idx)

    state = np.reshape(state, (5, 390))
    state = torch.tensor(state, dtype=torch.float).unsqueeze(0).unsqueeze(0).to("cuda")

    return state


def _find_trades(df, params, version):
    __VERSION = 3
    version.clear()
    version.append(__VERSION)

    filter_mode_on = 'filter_sym' in params.keys() and 'filter_date' in params.keys()

    entries = []
    exits = []

    date = params['date']

    open_idx = cu.get_time_index(df, date, params['__chart_begin_hh'], params['__chart_begin_mm'], 0)
    close_idx = cu.get_time_index(df, date, params['__chart_end_hh'], params['__chart_end_mm'], 0)

    trading_start_idx = cu.get_time_index(df, date, params['trading_begin_hh'], params['trading_begin_mm'], 0)
    if trading_start_idx is None:
        trading_start_idx = df.index[0]

    model = params['nn_model']

    i = trading_start_idx
    while i < close_idx:
        close = df['Close'][i]  # eliminating search in dataframe ... maybe increases exectution speed
        with torch.no_grad():
            data = gen_state(df, i, open_idx)

            buy_output = model(data)
            res = buy_output.max(1)[1].view(1, 1)
            predicted_label = res[0][0].to("cpu").numpy()

            if predicted_label == 5:
                buy_price = close

                print(params["symbol"], "BUY", df['Time'][i], buy_price, end="")

                stop_price = buy_price * (1 + params['stop'] / 100)
                params['stop_price'] = stop_price
                params['target_price'] = buy_price * (1 + params['target'] / 100)
                params['open_idx'] = open_idx

                entries.append([df['Time'][i], buy_price, [], params['stop']])

                sell_time, sell_price, exit_type, sell_index = _find_exit(df, i, params)

                exits.append([sell_time, sell_price, exit_type, sell_index])
                print(">>>>", i, sell_index)

                # i = sell_index

        if filter_mode_on:
            print('  ', df['Time'][i].time())

        i = i + 1

    return entries, exits


def _find_exit(df, entry_index, params):
    exit_type = None
    sold = False
    sell_price = None

    entry_price = df['Close'][entry_index]
    chart_end_idx = df.index[-1]
    open_idx = params['open_idx']

    model = params['nn_model']

    j = entry_index + 1

    while j < chart_end_idx and not sold:
        if df['Low'][j] < params['stop_price']:
            sold = True
            sell_price = params['stop_price']
            exit_type = "STOP"
            print("  STOP", df['Time'][j], "%.2f" % sell_price, "%  stop:", str(params['stop']) + "%", end="")
        else:
            with torch.no_grad():
                data = gen_state(df, j, open_idx)

                buy_output = model(data)
                res = buy_output.max(1)[1].view(1, 1)
                predicted_label = res[0][0].to("cpu").numpy()

                if predicted_label <= 1:
                    sold = True
                    sell_price = df.loc[j]['Close']
                    exit_type = "SELL"
                    print("  SELL", df['Time'][j], "%.2f" % sell_price, end="")
        '''
        elif df['High'][j] > params['target_price']:            
            sold = True
            sell_price = params['target_price']
            exit_type = "SELL"
            print("  SELL", df['Time'][j], "%.2f" % sell_price, end="")
        '''
        j = j + 1

    if not sold:
        if j == chart_end_idx:
            sell_price = df.loc[j]['Close']
            exit_type = "Timeout"
            print("  Timeout: ", df['Time'][j], "%.2f" % sell_price, "j:", j, "idx_end: ", chart_end_idx, end="")

        else:
            print("")
            print("idx_end: ", chart_end_idx, "time:", df.loc[j]["Time"], "j:", j)
            print(colored("Algorithm ERROR!!!!", color='red'))

    gain = int(100 * (sell_price - entry_price) / entry_price)
    print(colored(" " + str(gain) + "%", color="red" if gain < 0 else "green"))

    return df.loc[j - 1]['Time'], sell_price, exit_type, j - 1


def get_pattern_file_path(params):
    return "trades\\" + params['pattern'] + "\\" + \
           str(params['pattern']) + "_" + \
           "V" + str(params['version']) + "_" + \
           ".csv"


def show_day_distribution(params):
    params['pattern'] = 'ai'
    params['version'] = 3
    df = pd.read_csv(get_pattern_file_path(params))
    df.date = pd.to_datetime(df.date, format="%Y-%m-%d")

    df.sort_values(by='date')

    n = len(df)
    gain = []
    for i in range(0, n):
        gain.append(100 * (df["sell_price"][i] / df["entry_price"][i] - 1))

    print(len(df))
    # plt.hist(df['date'], bins=1000, alpha=0.5, align='mid', rwidth=4)
    plt.scatter(x=df["date"], y=gain)

    plt.grid(True, which='major')
    plt.show()


def init_ai(params):
    model = Net().to("cuda")

    path = params['model_path']
    if os.path.isfile(path):
        model.load_state_dict(torch.load(path))
        print(colored("Loaded AI state file: " + path, color="green"))
    else:
        print(colored("Could not find AI state file: " + path, color="red"))

    model.eval()
    params['nn_model'] = model

    return params


def main():
    params = get_default_params()

    # params['filter_sym'] = 'KBSF'
    # params['filter_date'] = '2017-11-09'

    params = init_ai(params)
    search_patterns(params)

    show_day_distribution(params)
    ####################################################
    # Simulation

    filter_mode_on = 'filter_sym' in params.keys() and 'filter_date' in params.keys()
    if not filter_mode_on:
        simulate_pattern(params)


def get_default_params():
    params = {
        '__chart_begin_hh': 9,
        '__chart_begin_mm': 30,
        '__chart_end_hh': 15,
        '__chart_end_mm': 59,

        'trading_begin_hh': 9,
        'trading_begin_mm': 40,
        'last_entry_hh': 15,
        'last_entry_mm': 45,

        'stop': -5,
        'target': 10,

        'no_charts': True,

        'chart_list_file': "data\\active_days_all.csv",

        'model_path': "checkpoints\\checkpoint_700",
        'split_train_test': 0.91
    }

    return params


if __name__ == "__main__":
    main()
