import numpy as np
import common as cu
import pandas as pd

DATA_ROWS = 5
DAY_IN_MINUTES = 390
DAILY_CHART_LENGTH = 122

EXTENDED_CHART_LENGTH = DAILY_CHART_LENGTH + DAY_IN_MINUTES
LABEL_SIZE = 1

EXT_DATA_SIZE = EXTENDED_CHART_LENGTH * DATA_ROWS + LABEL_SIZE


def save_state_chart(state, t, symbol, date, idx, length):
    s = state.reshape(DATA_ROWS, length)

    o = s[0]
    c = s[1]
    h = s[2]
    l = s[3]
    v = s[4]
    t = t[:length]

    print("len(o):", len(o), "len(t):", len(t))

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


def create_padded_state_vector(df, entry_idx, open_idx, debug=False):
    """ idx - is the candle index in range 0 ... 390 """

    t = df.Time.to_list()
    o = df.Open.to_list()
    c = df.Close.to_list()
    h = df.High.to_list()
    l = df.Low.to_list()
    v = df.Volume.to_list()

    idx = entry_idx - open_idx
    if debug:
        print("calc_normalized_state")
        print('open_idx:', open_idx, 'entry_idx', entry_idx, 'idx:', idx)
        print('full len(o)', len(o))
        print('full len(v)', len(v))

    t = t[:idx + 1]
    o = o[:idx + 1]
    c = c[:idx + 1]
    h = h[:idx + 1]
    l = l[:idx + 1]
    v = v[:idx + 1]

    if debug:
        print('entry len(o)', len(o))
        print('entry len(v)', len(v))
        print('o:', len(o), " >>> ", o)
        print('entry:', t[idx])

    price = np.concatenate((o, c, h, l))
    price = cu.scale_to_1(price)
    price = price.reshape(4, idx + 1)

    o = price[0]
    c = price[1]
    h = price[2]
    l = price[3]
    v = cu.scale_to_1(np.array(v))

    padding_size = DAY_IN_MINUTES - (idx + 1)
    padding = [0] * padding_size

    if debug:
        print('padding_size:', padding_size)

    o = np.concatenate((padding, o))
    c = np.concatenate((padding, c))
    h = np.concatenate((padding, h))
    l = np.concatenate((padding, l))
    v = np.concatenate((padding, v))

    if debug:
        print('padded len(o)', len(o))
        print('padded len(v)', len(v))

    state = np.concatenate((o, c, h, l, v))

    return state


def create_state_vector(df_history, df, entry_idx, open_idx, debug=False):
    """ idx - is the candle index in range 0 ... 390 """

    ##################### SCALING INTRA-DAY ##############################

    t = df.Time.to_list()
    o = df.Open.to_list()
    c = df.Close.to_list()
    h = df.High.to_list()
    l = df.Low.to_list()
    v = df.Volume.to_list()

    idx = entry_idx - open_idx
    if debug:
        print("calc_normalized_state")
        print('open_idx:', open_idx, 'entry_idx', entry_idx, 'idx:', idx)
        print('full len(o)', len(o))
        print('full len(v)', len(v))

    o = o[:idx + 1]
    c = c[:idx + 1]
    h = h[:idx + 1]
    l = l[:idx + 1]
    v = v[:idx + 1]

    if debug:
        t = t[:idx + 1]
        print('entry len(o)', len(o))
        print('entry len(v)', len(v))
        print('o:', len(o), type(o), " >>> ", o)
        print('entry:', t[idx])

    price = np.concatenate((o, c, h, l))
    price = cu.scale_to_1(price)
    price = price.reshape(4, idx + 1)

    o = price[0]
    c = price[1]
    h = price[2]
    l = price[3]
    v = cu.scale_to_1(np.array(v))

    if debug:
        print('normaliazed o:', type(o))
        print('normaliazed o shape:', o.shape)
        print('normaliazed len(o):', len(o))

    ##################### PADDING INTRA-DAY ##############################

    padding_size = DAY_IN_MINUTES - (idx + 1)
    padding = [0] * padding_size

    if debug:
        print('padding_size:', padding_size)

    o = np.concatenate((padding, o))
    c = np.concatenate((padding, c))
    h = np.concatenate((padding, h))
    l = np.concatenate((padding, l))
    v = np.concatenate((padding, v))

    if debug:
        print('padded len(o)', len(o))
        print('padded len(v)', len(v))

    ##################### HISTORY SCALING ##############################

    ho = df_history.Open.to_list()
    hc = df_history.Close.to_list()
    hh = df_history.High.to_list()
    hl = df_history.Low.to_list()
    hv = df_history.Volume.to_list()

    if debug:
        print("ho:", ho)
        print('full len(ho)', len(ho))
        print('full len(hv)', len(hv))

    price = np.concatenate((ho, hc, hh, hl))
    price = cu.scale_to_1(price)
    price = price.reshape(4, len(ho))

    ho = price[0]
    hc = price[1]
    hh = price[2]
    hl = price[3]
    hv = cu.scale_to_1(np.array(hv))

    if debug:
        print('normalized len(ho)', len(ho))
        print('normalized len(hv)', len(hv))

    ##################### HISTORY PADDING ##############################

    padding_size = DAILY_CHART_LENGTH - len(df_history)
    padding = [0] * padding_size

    if debug:
        print('history padding_size:', padding_size)

    if padding_size > 0:
        ho = np.concatenate((padding, ho))
        hc = np.concatenate((padding, hc))
        hh = np.concatenate((padding, hh))
        hl = np.concatenate((padding, hl))
        hv = np.concatenate((padding, hv))

    if debug:
        print('padded len(o)', len(ho))
        print('padded len(v)', len(hv))

    ##################### APPEND HISTORY ###############################

    o = np.concatenate((ho, o))
    c = np.concatenate((hc, c))
    h = np.concatenate((hh, h))
    l = np.concatenate((hl, l))
    v = np.concatenate((hv, v))

    state = np.concatenate((o, c, h, l, v))

    if debug:
        print('total len(o)', len(o))
        print('total len(v)', len(v))
        print('len(state)', len(state))

    return state


def pad_state_to_512(state):
    state = state.reshape(DATA_ROWS, DAY_IN_MINUTES)
    padding = np.zeros((DATA_ROWS, 512 - DAY_IN_MINUTES), dtype=float)
    return np.concatenate((padding, state), axis=1)


def update_candle(df, params):
    idx = params['date_index']

    df.at[idx, 'Open'] = params['Open']
    df.at[idx, 'Close'] = params['Close']
    df.at[idx, 'Low'] = params['Low']
    df.at[idx, 'High'] = params['High']
    df.at[idx, 'Volume'] = params['Volume']

    return df
