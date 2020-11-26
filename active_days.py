import os
import common
import queue
from termcolor import colored
import threading
import time
import pandas as pd
import multiprocessing as mp
import numpy as np


__active_days_file = "data\\active_days_ext.csv"


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

            if gap > params['min_gap_up'] and df.loc[i, 'Open'] > params['day_open_above'] and (df.loc[i, 'Volume'] > params['min_gap_up_volume']):
                t = df["Time"][df.index[i]].strftime("%Y-%m-%d")
                print(symbol, t)
                relevant_days_list.append([gap, t, symbol])
            elif range_ > params['min_range'] and df.loc[i, 'High'] > params['day_high_above'] and (df.loc[i, 'Volume'] > params['min_range_volume']):
                t = df["Time"][df.index[i]].strftime("%Y-%m-%d")
                print(symbol, t, "     <---")
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


def get_default_params():
    params = {
        'day_high_above': 1,
        'day_open_above': 1,
        'min_gap_up': 10,
        'min_gap_up_volume': 1000000,
        'min_range': 20,
        'min_range_volume': 20000000,
    }

    return params


def generate_active_days_file():
    movers = find_active_days_mp(get_default_params(), 16)

    print(movers)

    data = np.array(movers)
    df = pd.DataFrame(data=data, columns=['gap', 'date', 'symbol'])

    common.make_dir(common.__data_dir)
    df.to_csv(__active_days_file)


def _main():
    generate_active_days_file()


if __name__ == "__main__":
    _main()
