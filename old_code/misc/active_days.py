import os
import common
import queue
from termcolor import colored
import threading
import time
import pandas as pd
import multiprocessing as mp
import numpy as np


def _get_active_days_for(symbol, params):
    df = common.get_daily_chart_for(symbol)
    df.Time = pd.to_datetime(df.Time, format="%Y-%m-%d")

    relevant_days_list = []
    n = len(df)

    for i in range(1, n):
        if df.loc[i - 1]['Close'] > 0 and df.loc[i]['Low'] > 0:

            gap = (df.loc[i]['Open'] - df.loc[i - 1]['Close']) / df.loc[i-1]['Close']
            gap = round(gap * 100)  # gap up in percents

            range_ = (df.loc[i]['High'] - df.loc[i]['Low']) / df.loc[i]['Low']
            range_ = round(range_ * 100)  # in percents

            #####################################################################
            #  DEBUG
            '''
            xxx = df["Time"][df.index[i]]
            if xxx == "2019-05-01":
                print("Gap: " + str(gap))
                print("Range: " + str(range_))
                print(df.loc[i, 'Volume'])
            '''
            #####################################################################

            if df.loc[i, 'Volume'] * df.loc[i, 'Close'] > params['min_value_traded']:
                if df.loc[i, 'Open'] > params['day_open_above'] and df.loc[i, 'High'] > params['day_high_above']:
                    if params['min_gap_up'] is not None:
                        if gap > params['min_gap_up']:
                            t = df["Time"][df.index[i]].strftime("%Y-%m-%d")
                            print(symbol, t, "    gap:", gap)
                            relevant_days_list.append([gap, t, symbol])

                    if params['min_range'] is not None:
                        if range_ > params['min_range']:
                            t = df["Time"][df.index[i]].strftime("%Y-%m-%d")
                            print(symbol, t, "     <--- range:", range_)
                            relevant_days_list.append([range_, t, symbol])

    print(symbol + " Relevant Days (Gap-Ups + Big movers): " + str(int(len(relevant_days_list))))
    return relevant_days_list


def find_active_days_mp(params, cpu_count=1):
    symbols = common.get_list_of_symbols_with_daily_chart()

    results = []
    if cpu_count > 1:
        print("Num CPUs: ", cpu_count)

        pool = mp.Pool(cpu_count)
        mp_results = [pool.apply_async(_get_active_days_for, args=(symbol, params)) for symbol in symbols]
        pool.close()
        pool.join()

        for res in mp_results:
            movers = res.get(timeout=1)
            if len(movers) > 0:
                for m in movers:
                    results.append(m)

    else:
        print("Single CPU execution")

        for symbol in symbols:
            movers = _get_active_days_for(symbol, params)
            if len(movers) > 0:
                for m in movers:
                    results.append(m)

    return results


def find_all_days(params):
    symbols = common.get_list_of_symbols_with_daily_chart()

    results = []
    for symbol in symbols:
        files = common.get_intraday_chart_files_for(symbol)

        for file in files:
            date = file.replace(".csv", "")[-8:]

            df = common.get_intraday_chart_for(symbol, date)
            t_ = df.Time.to_list()
            price = np.array(df.Close.to_list())
            volume = np.array(df.Volume.to_list())

            val = sum(price * volume)

            print(file + date + "   %sM" % int(val / 10000000), end="")

            if val > params['min_value_traded']:
                results.append([date, symbol])
                print("  <----")
            else:
                print("")

    return results


def get_default_params():
    params = {
        'min_value_traded': 10000000,

        'day_high_above': 10,
        'day_open_above': 1,

        'min_gap_up': 5,
        'min_gap_up_volume': 1,
        'min_range': None,
        'min_range_volume': None,

        'file_path': "data\\active_days_all.csv"
    }

    return params


def generate_active_days_file():
    params = get_default_params()
    # movers = find_active_days_mp(params, cpu_count=10)
    movers = find_all_days(params)

    print(movers)

    if len(movers) > 0:
        data = np.array(movers)
        df = pd.DataFrame(data=data, columns=['date', 'symbol'])

        common.make_dir(common.__data_dir)
        df.to_csv(params['file_path'])


def _main():
    generate_active_days_file()


if __name__ == "__main__":
    _main()
