import pandas as pd
from chart import Trade_Env
import numpy as np
from chart import create_padded_state_vector
from chart import DAY_IN_MINUTES
import common as cu


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

    bl = [0] * 3
    state = calc_normalized_state(_open, _close, _high, _low, vol, bl, 2, debug=True)
    state = state.reshape(7, DAY_IN_MINUTES)
    print(state)


def test2():
    symbol = "AAL"
    date = "2020-02-26"
    df = cu.get_chart_data_prepared_for_ai(symbol, date)

    idx = 0

    o = df.Open.to_list()[:idx + 1]
    c = df.Close.to_list()[:idx + 1]
    h = df.High.to_list()[:idx + 1]
    l = df.Low.to_list()[:idx + 1]
    v = df.Volume.to_list()[:idx + 1]
    t = df.Time.to_list()

    print('Volume:', v)

    bl = [0] * (idx + 1)

    state = calc_normalized_state(o, c, h, l, v, bl, idx, debug=True)
    print("len(state): ", len(state))
    print(state[0:5])

    cu.show_1min_chart(df, symbol, date, "", [], [], [], [], None)

    state = state.reshape(7, DAY_IN_MINUTES)
    o = state[0]
    c = state[1]
    h = state[2]
    l = state[3]
    v = state[4]

    print(v)

    dx = pd.DataFrame({
        'Time': t,
        'Open': o,
        'Close': c,
        'High': h,
        'Low': l,
        'Volume': v})

    cu.show_1min_chart(dx, idx, symbol, date, "", [], [], [], [], None)


def test4():
    movers = pd.read_csv('data\\active_days.csv')
    env = Trade_Env(movers, simulation_mode=True, sim_chart_index=2300)

    env.reset(debug=True)
    s, _, _ = env.step(2, debug=True)
    s, _, _ = env.step(0, debug=True)
    s, _, _ = env.step(0, debug=True)
    s, _, _ = env.step(0, debug=True)
    s, _, _ = env.step(2, debug=True)
    s, _, _ = env.step(1, debug=True)
    s, _, _ = env.step(1, debug=True)
    s, _, _ = env.step(1, debug=True)
    s, _, _ = env.step(1, debug=True)
    s, _, _ = env.step(0, debug=True)

    for i in range(390):
        s, _, _ = env.step(0, debug=False)

    s = s.to("cpu")
    print(s.shape)

    env.save_normalized_chart()


def test5():
    movers = pd.read_csv('data\\active_days.csv')
    env = Trade_Env(movers, simulation_mode=True)

    s, _, _ = env.step(0, debug=True)

    z = env.reset(debug=True)
    print(z)
    for i in range(0, 395):
        print("-------------", i)
        s, _, _ = env.step(0, debug=True)
        s = s.to("cpu")
        print(s.shape)
        z = np.reshape(s, (6, 390))
        print(z)

    env.save_normalized_chart()


def test6():
    x = [0, 5, 12, 0, 1, -3]
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = []
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = [0, 5, 5, 5, 5, 10]
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = [5, 5, 5, 5, 5, 5]
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = [5]
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = [0]
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = [-5, -5]
    print(x, " => ", cu.normalize_middle(np.array(x)))

    x = [-5, -15, 0, 0, 0, -20]
    print(x, " => ", cu.normalize_middle(np.array(x)))


def test7():
    x = [0, 5, 12, 0, 1, -3]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = []
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = [0, 5, 5, 5, 5, 10]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = [5, 5, 5, 5, 5, 5]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = [5]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = [0]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = [-5, -5]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    x = [-5, -15, 0, 0, 0, -20]
    y = cu.normalize_middle(np.array(x))
    print(x, " => ", y)
    print(cu.shift_and_scale(y), "\n")

    z = -0.65
    print(z, " ----> ", cu.shift_and_scale([z]), "\n")
    z = [-0.65]
    print(z, " ----> ", cu.shift_and_scale(z), "\n")
    z = [0]
    print(z, " ----> ", cu.shift_and_scale(z), "\n")
    z = [1]
    print(z, " ----> ", cu.shift_and_scale(z), "\n")
    z = [-1]
    print(z, " ----> ", cu.shift_and_scale(z), "\n")


def test_dot_prod():
    a = np.array([1, 2, 3, 4, 5, 6])
    b = np.array([2, 0, 2, 0, 2, 0])

    c = a * b
    print(c)


test_dot_prod()



