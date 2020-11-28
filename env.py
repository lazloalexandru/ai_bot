import numpy as np
from termcolor import colored
import common as cu
import random
import torch
import pandas as pd

DATA_ROWS = 7
DAY_IN_MINUTES = 390


def calc_normalized_state(o, c, h, l, v, b, idx, debug=False):
    """ idx - is the candle index in range 0 ... 390 """

    price = np.concatenate((o, c, h, l))
    price = cu.normalize(price)

    n = len(o)
    price = price.reshape(4, n)
    o = price[0]
    c = price[1]
    h = price[2]
    l = price[3]

    t = []
    for i in range(0, idx+1):
        t.append(i / DAY_IN_MINUTES)

    v = cu.normalize(np.array(v))

    padding_size = DAY_IN_MINUTES - (idx + 1)
    padding = [0] * padding_size

    if debug:
        print("calc_normalized_state")
        print('len(t):', len(t))
        print('len(o)', len(o))
        print('len(v)', len(v))
        print('len(b)', len(b))
        print('padding_size:', padding_size)

    o = np.concatenate((padding, o))
    c = np.concatenate((padding, c))
    h = np.concatenate((padding, h))
    l = np.concatenate((padding, l))
    v = np.concatenate((padding, v))
    t = np.concatenate((padding, t))
    b = np.concatenate((padding, b))

    if debug:
        print('padded len(t):', len(t))
        print('padded len(o)', len(o))
        print('padded len(v)', len(v))
        print('padded len(b)', len(b))

    state = np.concatenate((o, c, h, l, v, t, b))

    return state


class Trade_Env:
    def __init__(self, movers, simulation_mode, sim_chart_index=None):
        self.sim_chart_index = sim_chart_index

        self.movers = movers
        self.num_movers = len(movers)

        self.buy_prices = []
        self.num_trades = 0
        self.buy_locations_vector = [0]
        self.buy_indexes = []

        self.df = None
        self.symbol = None
        self.date = None

        self.idx = 0

        self._open = None
        self._close = None
        self._high = None
        self._low = None
        self._volume = None
        self._time = None   # used for debugging only

        self._normalized_close = []

        self._state = None
        self._render = False

        self.entries = []
        self.exits = []

        self.simulation_mode = simulation_mode

        self.reset()

    def _init_normalised_clsose(self, debug):
        o = self._open
        c = self._close
        h = self._high
        l = self._low
        v = self._volume
        b = [0] * DAY_IN_MINUTES

        s = calc_normalized_state(o, c, h, l, v, b, DAY_IN_MINUTES-1, debug)
        s = np.reshape(s, (DATA_ROWS, DAY_IN_MINUTES))

        self._normalized_close = s[1]

    def reset(self, debug=False):
        self._pick_chart()
        self._open = self.df.Open.to_list()
        self._close = self.df.Close.to_list()
        self._high = self.df.High.to_list()
        self._low = self.df.Low.to_list()
        self._volume = self.df.Volume.to_list()
        self._time = self.df.Time.to_list()

        self._init_normalised_clsose(debug)

        self.idx = 0

        self.entries = []
        self.exits = []

        self.buy_locations_vector = [0]
        self.buy_indexes = []
        self.buy_prices = []
        self.num_trades = 0

        if debug:
            print("RESET", self.idx, self._time[-1], self._time[self.idx], len(self.buy_locations_vector))

        self._calc_state(debug)

        return self._state

    def _calc_state(self, debug):
        _time = self._time[:self.idx]

        o = self._open[:self.idx + 1]
        c = self._close[:self.idx + 1]
        h = self._high[:self.idx + 1]
        l = self._low[:self.idx + 1]
        v = self._volume[:self.idx + 1]

        if debug:
            print("calc_state len(o)", len(o), "idx:", self.idx)

        s = calc_normalized_state(o, c, h, l, v, self.buy_locations_vector, self.idx, debug)
        s = np.reshape(s, (DATA_ROWS, DAY_IN_MINUTES))

        self._state = torch.tensor(s, dtype=torch.float).unsqueeze(0).unsqueeze(0).to("cuda")

        # print(self.idx, self.close_idx, self._time[-1], self._time[self.idx], self.buy_locations_vector)
        # print(self.idx, self.buy_locations_vector, self.buy_indexes)

    def step(self, action, debug=False):
        if debug:
            print("STEP", self.idx, self._time[-1], self._time[self.idx], len(self.buy_locations_vector))

        done = False
        reward = 0

        num_positions = len(self.buy_prices)
        if self.idx == DAY_IN_MINUTES - 1:
            if num_positions > 0:
                self.num_trades += num_positions

                avg_price = sum(self.buy_prices) / num_positions
                sell_price = self._normalized_close[self.idx]

                self.buy_prices = self.buy_prices + [1] * len(self.buy_prices)
                self.buy_prices *= 10
                ap = sum(self.buy_prices) / num_positions
                sp = 10*(sell_price + 1)
                reward = 100 * (sp / ap - 1)

                unit_gain = reward
                reward = reward * num_positions

                # Close all positions
                self.buy_locations_vector = [0] * DAY_IN_MINUTES
                self.buy_prices = []
                self.buy_indexes = []

                if self.simulation_mode:
                    c = "green" if reward > 0 else "red"
                    self.exits.append([self._time[self.idx], sell_price])
                    print(colored("%s  BUY: %.2f (%.2f)" % (self.symbol, avg_price, ap), color=c), end="")
                    print(colored("   SELL %.2f (%.2f)   REWARD: %.2f x %s => %.2f" % (sell_price, sp, unit_gain, num_positions, reward), color=c))
                else:
                    print(colored("XXX", color='red'), end="")

            done = True
            print("  Trades:", self.num_trades)
        elif self.idx < DAY_IN_MINUTES:
            if action == 0:  # BUY
                buy_price = self._normalized_close[self.idx]
                self.buy_prices.append(buy_price)
                buy_index = len(self.buy_locations_vector)-1
                self.buy_indexes.append(buy_index)
                self.buy_locations_vector[buy_index] = 1

                if self.simulation_mode:
                    self.entries.append([self._time[self.idx], buy_price])
                else:
                    print(".", end="")
            elif action == 1:  # SELL
                if num_positions > 0:
                    self.num_trades += 1

                    buy_price = self.buy_prices.pop()
                    buy_index = self.buy_indexes.pop()
                    self.buy_locations_vector[buy_index] = 0

                    sell_price = self._normalized_close[self.idx]

                    bp = 10 * (buy_price + 1)
                    sp = 10 * (sell_price + 1)
                    reward = 100 * (sp / bp - 1)

                    if self.simulation_mode:
                        self.exits.append([self._time[self.idx], sell_price])

                        c = "green" if reward > 0 else "red"
                        print(colored("%s  BUY: %.2f (%.2f)" % (self.symbol, buy_price, bp), color=c), end="")
                        print(colored("   SELL %.2f (%.2f)   REWARD: %.2f" % (sell_price, sp, reward), color=c))
                    else:
                        print("|", end="")

            elif action == 2:  # IDLE
                reward = 0

        if self.idx < DAY_IN_MINUTES - 1:
            self.buy_locations_vector.append(0)
            self.idx += 1

        self._calc_state(debug)
        return self._state, reward, done

    def _pick_chart(self):
        open_idx = None
        close_idx = None

        while (open_idx is None) or (close_idx is None):
            if self.simulation_mode:
                if self.sim_chart_index is None:
                    rand_idx = random.randint(int(self.num_movers * 0.8) + 1, self.num_movers - 1)
                else:
                    rand_idx = self.sim_chart_index
            else:
                if self.sim_chart_index is None:
                    rand_idx = random.randint(0, int(self.num_movers * 0.8))
                else:
                    rand_idx = self.sim_chart_index

            self.symbol = self.movers.iloc[rand_idx]["symbol"]
            self.date = self.movers.iloc[rand_idx]["date"]

            self.df = cu.get_chart_data_prepared_for_ai(self.symbol, self.date)

            print(self.symbol, self.date)

            if self.df is not None:
                open_idx = cu.get_time_index(self.df, self.date, 9, 30, 0)
                close_idx = cu.get_time_index(self.df, self.date, 15, 59, 0)

    def save_chart(self, filename=None):
        # if len(self.entries) > 0:
        if True:
            print("save_chart -> len(time)", len(self._time), len(self.entries))

            state = self._state.reshape(7, DAY_IN_MINUTES)
            s = state.to(torch.device("cpu")).numpy()

            o = s[0]
            c = s[1]
            h = s[2]
            l = s[3]
            v = s[4]
            t = self._time[0:self.idx+1]

            from_idx = -(self.idx + 1)
            o = o[from_idx:]
            c = c[from_idx:]
            h = h[from_idx:]
            l = l[from_idx:]
            v = v[from_idx:]
            print("save_chart   ->  idx:", self.idx, len(o), len(t))

            dx = pd.DataFrame({
                'Time': t,
                'Open': o,
                'Close': c,
                'High': h,
                'Low': l,
                'Volume': v})

            filename = self.symbol
            images_dir_path = "trades\\"
            cu.show_1min_chart_normalized(dx,
                                          self.idx,
                                          self.symbol,
                                          self.date,
                                          "",
                                          self.entries,
                                          self.exits,
                                          [],
                                          [],
                                          images_dir_path,
                                          filename)

    @property
    def num_states(self):
        return len(self._state)

    @property
    def num_actions(self):
        return 3

    @property
    def state(self):
        return self._state

    @property
    def state_shape(self):
        return self._state.shape
