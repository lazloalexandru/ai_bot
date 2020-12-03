import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import common as cu

__input_dir = "data\\intraday_charts"
__output_dir = cu.__normalized_states_root_dir + "\\intraday_charts"


def test1():
    _open = [3, 3, 3]
    _close = [4, 4, 4]
    _high = [6, 6, 6]
    _low = [2, 2, 2]
    t1 = pd.to_datetime("2020-11-20 9:30:00", format="%Y-%m-%d  %H:%M:%S")
    t2 = pd.to_datetime("2020-11-20 10:00:00", format="%Y-%m-%d  %H:%M:%S")
    t3 = pd.to_datetime("2020-11-20 16:00:00", format="%Y-%m-%d  %H:%M:%S")
    _time = [t1, t2, t3]
    vol = [2000, 4000, 6000]

    state = cu.normalize_ochl_v_t_state(_open, _close, _high, _low, vol, 2)

    state.astype('float').tofile("data\\test.dat")
    print(state)
    zzz = np.fromfile("data\\test.dat")
    print(zzz)


def test2():
    symbol = "AAL"
    date = "2020-02-26"
    df = cu.get_chart_data_prepared_for_ai(symbol, date)

    o = df.Open.to_list()
    c = df.Close.to_list()
    h = df.High.to_list()
    l = df.Low.to_list()
    v = df.Volume.to_list()
    t = df.Time.to_list()

    state = cu.normalize_ochl_v_t_state(o, c, h, l, v, 390)
    state.astype('float').tofile("data\\test.dat")
    zzz = np.fromfile("data\\test.dat")
    print(len(zzz))
    print(zzz[0:5])

    cu.show_1min_chart(df, symbol, date, "", [], [], [], [], None)

    print(len(state))
    state = zzz[:-1]
    print(len(state))

    state = state.reshape(5, cu.DAY_IN_MINUTES)
    o = state[0]
    c = state[1]
    h = state[2]
    l = state[3]
    v = state[4]

    dx = pd.DataFrame({
        'Time': t,
        'Open': o,
        'Close': c,
        'High': h,
        'Low': l,
        'Volume': v})

    cu.show_1min_chart(dx, symbol, date, "", [], [], [], [], None)


def test3(symbol, date, idx):

    df = cu.get_chart_data_prepared_for_ai(symbol, date)

    states = cu.get_normalized_states(symbol, date)
    state = states[idx]
    print(len(state))
    state = state[:-1]
    print(len(state))

    state = state.reshape(5, cu.DAY_IN_MINUTES)

    o = state[0]
    c = state[1]
    h = state[2]
    l = state[3]
    v = state[4]

    dx = pd.DataFrame({
        'Time': df.Time,
        'Open': o,
        'Close': c,
        'High': h,
        'Low': l,
        'Volume': v})

    cu.show_1min_chart(dx, symbol, date, "", [], [], [], [], None)


def show_state(state, t, symbol, date):
    state = state[:-1]

    state = state.reshape(5, cu.DAY_IN_MINUTES)
    o = state[0]
    c = state[1]
    h = state[2]
    l = state[3]
    v = state[4]

    dx = pd.DataFrame({
        'Time': t,
        'Open': o,
        'Close': c,
        'High': h,
        'Low': l,
        'Volume': v})

    cu.show_1min_chart(dx, symbol, date, "", [], [], [], [], None)


def _gen_state_for_idx(df, idx, open_idx):
    o = df.Open.to_list()[:idx + 1]
    c = df.Close.to_list()[:idx + 1]
    h = df.High.to_list()[:idx + 1]
    l = df.Low.to_list()[:idx + 1]
    v = df.Volume.to_list()[:idx + 1]
    t = df.Time.to_list()[:idx + 1]

    state = cu.normalize_ochl_v_t_state(o, c, h, l, v, idx - open_idx + 1)

    return state


def _generate_normalized_states(symbol, date):
    print(symbol, date)

    xd = str(date).replace("-", "")
    result_file = __output_dir + "\\" + symbol + "\\" + symbol + '_' + xd + ".dat"
    result_dir = __output_dir + "\\" + symbol

    if not os.path.isdir(result_dir):
        try:
            os.mkdir(result_dir)
        except OSError:
            print("Creation of the directory %s failed" % result_dir, flush=True)

    if os.path.isfile(result_file):
        print(result_file, " Already generated.", flush=True)
    else:
        df = cu.get_chart_data_prepared_for_ai(symbol, date)

        if df is not None:
            open_idx = cu.get_time_index(df, date, 9, 30, 0)
            close_idx = cu.get_time_index(df, date, 15, 59, 0)

            if open_idx is not None and close_idx is not None:
                states = []

                idx = open_idx
                while idx <= close_idx:
                    states.append(_gen_state_for_idx(df, idx, open_idx))
                    idx += 1

                byte_data = np.concatenate(states)
                byte_data.astype('float').tofile(result_file)

                '''  DEBUG
                print(len(byte_data), byte_data.shape, result_file) 
                zzz = np.fromfile(result_file)
                zzz = zzz.reshape(cu.DAY_IN_MINUTES, 1951)
                show_state(zzz[300], df.Time.to_list(), symbol, date)
                '''
    return 1


def generate_normalized_charts(num_cpu=None):
    cu.make_dir(cu.__normalized_states_root_dir)
    cu.make_dir(__output_dir)

    print("Creating list of intraday chart files ... ")

    params = []

    symbols = cu.get_list_of_symbols_with_intraday_chart()
    for symbol in symbols:
        dir_path = __input_dir + "\\" + symbol
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith(".csv"):
                    date = file.replace(".csv", "")[-8:]
                    params.append([symbol, date])

    if len(symbols) > 0:
        if num_cpu is None:
            num_cpu = mp.cpu_count()

        print("\nNum CPUs: ", num_cpu)

        ##################################################################
        # Used for debugging Single process ... to see errors
        '''
        for p in params:
            _generate_normalized_states(params[0][0], params[0][1])
        '''
        ###################################################################

        pool = mp.Pool(mp.cpu_count())
        for param in params:
            pool.apply_async(_generate_normalized_states, args=(param[0], param[1]))

        pool.close()
        pool.join()

        #########################################################

'''
def rename_files():
    symbols = cu.get_list_of_symbols_with_intraday_chart()
    for symbol in symbols:
        dir_path = cu.__normalized_states_dir + "\\" + symbol
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith(".csv"):
                    date = file.replace(".csv", "")[-8:]
                    path = dir_path + "\\" + file
                    print(path)
                    path_dat = path.replace(".csv", ".dat")
                    if os.path.isfile(path):
                        os.rename(path, path_dat)
'''

if __name__ == "__main__":
    generate_normalized_charts(12)
    # test3("AAL", "2020-02-26", 389)
    # states = cu.get_normalized_states("ALK", "2020-06-05")

