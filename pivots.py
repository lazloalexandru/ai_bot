
def get_index_of_min(df, start_idx, end_idx):
    mn = df['Low'][start_idx]
    idx = start_idx

    for i in range(start_idx+1, end_idx+1):
        if df['Low'][i] < mn:
            mn = df['Low'][i]
            idx = i

    # str(df.loc[idx]['Time'].time()), mx
    return idx, mn


def get_index_of_max(df, start_idx, end_idx):
    mx = df['High'][start_idx]
    idx = start_idx

    for i in range(start_idx+1, end_idx+1):
        if df['High'][i] > mx:
            mx = df['High'][i]
            idx = i

    # str(df.loc[idx]['Time'].time()), mx
    return idx, mx


def get_pivots(df, p):
    ms = "mav" + str(p["mavs_p"])
    ml = "mav" + str(p["mavl_p"])

    pivots = []
    start_idx = df.index[p['pivot_calc_start_index']]
    idx_prev = start_idx

    state_open = False
    if df[ms][start_idx] > df[ml][start_idx]:
        state_open = True

    n = len(df)

    for i in range(start_idx, n):
        prev_state = state_open

        if df[ms][i] > df[ml][i]:
            state_open = True
        elif df[ms][i] < df[ml][i]:
            state_open = False

        if prev_state and not state_open:
            max_idx, val = get_index_of_max(df, idx_prev, i)
            pivots.append([val, True, df['Time'][max_idx], max_idx])
            idx_prev = i
        elif not prev_state and state_open:
            min_idx, val = get_index_of_min(df, idx_prev, i)
            pivots.append([val, False, df['Time'][min_idx], min_idx])
            idx_prev = i

    return pivots
