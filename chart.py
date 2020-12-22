import numpy as np
import common as cu
import pandas as pd

DATA_ROWS = 2
DAY_IN_MINUTES = 390
DAILY_CHART_LENGTH = 0

EXTENDED_CHART_LENGTH = DAILY_CHART_LENGTH + DAY_IN_MINUTES
LABEL_SIZE = 1

EXT_DATA_SIZE = EXTENDED_CHART_LENGTH * DATA_ROWS + LABEL_SIZE


def save_state_chart(state, t, symbol, date, idx, length):
    s = state.reshape(DATA_ROWS, length)

    o = s[0] * 0.95
    c = s[0]
    h = s[0]
    l = s[0]
    v = s[1]
    t = t[:length]

    dx = pd.DataFrame({
        'Time': t,
        'Open': o,
        'Close': c,
        'High': h,
        'Low': l,
        'Volume': v})

    filename = symbol + "_" + date + "_" + str(idx)

    images_dir_path = "trades\\"
    cu.show_1min_chart_normalized(dx, idx, symbol, date, images_dir_path, filename)


def create_state_vector(df_history, df, entry_idx, open_idx, debug=False):
    """ idx - is the candle index in range 0 ... 390 """

    ##################### SCALING INTRA-DAY ##############################

    t = df.Time.to_list()
    c = df.Close.to_list()
    v = df.Volume.to_list()

    idx = entry_idx - open_idx
    if debug:
        print("calc_normalized_state")
        print('open_idx:', open_idx, 'entry_idx', entry_idx, 'idx:', idx)
        print('full len(v)', len(v))

    c = c[:idx + 1]
    v = v[:idx + 1]

    if debug:
        t = t[:idx + 1]
        print('entry len(c)', len(c))
        print('entry len(v)', len(v))
        print('c:', len(c), type(c), " >>> ", c)
        print('entry:', t[idx])

    c = cu.normalize_middle(np.array(c))
    v = cu.normalize_0_1(np.array(v))

    if debug:
        print('normaliazed c:', type(c), c)
        print('normaliazed v:', type(v), v)
        print('normaliazed o shape:', c.shape)
        print('normaliazed len(o):', len(c))

    ##################### PADDING INTRA-DAY ##############################

    padding_size = DAY_IN_MINUTES - (idx + 1)
    padding = [0] * padding_size

    if debug:
        print('padding_size:', padding_size)

    c = np.concatenate((padding, c))
    v = np.concatenate((padding, v))

    if debug:
        print('padded len(o)', len(c))
        print('padded len(v)', len(v))

    if df_history is not None:
        ##################### HISTORY SCALING ##############################

        hc = df_history.Close.to_list()
        hv = df_history.Volume.to_list()

        if debug:
            print("hc:", hc)
            print('full len(ho)', len(hc))
            print('full len(hv)', len(hv))

        hc = cu.normalize_middle(np.array(hc))
        hv = cu.normalize_0_1(np.array(hv))

        if debug:
            print('normalized len(hc)', len(hc))
            print('normalized len(hv)', len(hv))

        ##################### HISTORY PADDING ##############################

        padding_size = DAILY_CHART_LENGTH - len(df_history)
        padding = [0] * padding_size

        if debug:
            print('history padding_size:', padding_size)

        if padding_size > 0:
            hc = np.concatenate((padding, hc))
            hv = np.concatenate((padding, hv))

        if debug:
            print('padded len(c)', len(hc))
            print('padded len(v)', len(hv))

        ##################### APPEND HISTORY ###############################

        c = np.concatenate((hc, c))
        v = np.concatenate((hv, v))

    state = np.concatenate((c, v))

    if debug:
        print('total len(c)', len(c))
        print('total len(v)', len(v))
        print('len(state)', len(state))

    return state


def pad_state_to_512(state):
    state = state.reshape(DATA_ROWS, DAY_IN_MINUTES)
    padding = np.zeros((DATA_ROWS, 512 - DAY_IN_MINUTES), dtype=float)
    return np.concatenate((padding, state), axis=1)


def update_candle(df, params):
    idx = params['date_index']

    if df is not None and idx is not None:
        df.at[idx, 'Open'] = params['Open']
        df.at[idx, 'Close'] = params['Close']
        df.at[idx, 'Low'] = params['Low']
        df.at[idx, 'High'] = params['High']
        df.at[idx, 'Volume'] = params['Volume']

    return df
