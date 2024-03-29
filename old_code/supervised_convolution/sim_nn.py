from termcolor import colored
import multiprocessing as mp
import numpy as np
import pandas as pd
from chart import create_padded_state_vector
from chart import save_state_chart
import common as cu


def _gen_input(xxx, i):

    symbol = xxx["sym"]
    date = xxx["date"]
    buy_price = xxx["entry_price"]
    sell_price = xxx["sell_price"]
    exit_type = xxx["exit_type"]
    entry_time = pd.to_datetime(xxx["entry_time"], format="%Y-%m-%d  %H:%M:%S")

    df = cu.get_chart_data_prepared_for_ai(symbol, date)
    error = None
    result = None

    if df is not None:
        open_idx = cu.get_time_index(df, date, 9, 30, 0)
        close_idx = cu.get_time_index(df, date, 15, 59, 0)
        entry_idx = cu.get_time_index(df, date, entry_time.hour, entry_time.minute, 0)

        if open_idx is None or close_idx is None or entry_idx is None:
            if entry_idx:
                error = symbol + "_" + date + "_" + str(entry_idx)
        else:
            t = df.Time.to_list()
            o = df.Open.to_list()
            c = df.Close.to_list()
            h = df.High.to_list()
            l = df.Low.to_list()
            v = df.Volume.to_list()

            idx = entry_idx - open_idx

            state = create_padded_state_vector(
                o[:entry_idx + 1],
                c[:entry_idx + 1],
                h[:entry_idx + 1],
                l[:entry_idx + 1],
                v[:entry_idx + 1],
                idx)

            # print(state.shape)
            # save_state_chart(state, t, symbol, date, idx)

            gain = 100 * (sell_price / buy_price - 1)

            # print(symbol, df.loc[entry_idx]["Time"], " ", end="", flush=True)
            print(symbol, df.loc[entry_idx]["Time"], " ", end="", flush=True)
            # print(buy_price, sell_price, "%.2f" % gain, "%", end="")
            print("%.2f" % gain, "%   ", end="")

            label = 0

            if exit_type == "Timeout":
                print("BAD - Timeout", flush=True)
            elif gain > 5 and exit_type == "SELL":
                print("GOOD", flush=True)
                label = 1
            elif gain < 0 and exit_type == "STOP":
                print("BAD", flush=True)
            else:
                print(colored("ERROR XXXXXXXXXXXXXXXXXXX", color="red"), flush=True)

            result = np.concatenate((state, [label]))

    return result, error


def gen_inputs_mp():
    df_trades = pd.read_csv("data\\trades.csv")

    num_trades = len(df_trades)
    print("Number of Trades:", num_trades)

    num_cpu = mp.cpu_count()
    print("\nNum CPUs: ", num_cpu)

    results = []
    errors = []

    for i in range(140000, 140010):
        print("")
        dfr, error = _gen_input(df_trades.loc[i], i)

        if len(dfr) > 0:
            results.append(dfr)

        if error is not None:
            errors.append(error)

    print("Errors: ", len(errors))
    for e in errors:
        print(e)

    xxx = np.array(results)
    print(xxx.shape)


def test_training_data():
    zzz = np.fromfile('data\\dataset.dat', dtype='float')

    n = len(zzz)
    rows = int(n / 1951)

    zzz = zzz.reshape(rows, 1951)

    print(zzz.shape)
    print(zzz[0], zzz[0][-1])
    print(zzz[1], zzz[1][-1])

    state = zzz[1230][:-1]

    print("state: ", state.shape)
    symbol = "AAL"
    date = "2020-04-13"
    df = cu.get_chart_data_prepared_for_ai(symbol, date)
    t = df.Time.to_list()
    save_state_chart(state, t, "----", date, 1)


if __name__ == '__main__':
    test_training_data()

