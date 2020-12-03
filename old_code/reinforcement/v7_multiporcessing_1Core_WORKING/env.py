import numpy as np
from termcolor import colored
import common as cu
import random
import torch
import pandas as pd

DATA_ROWS = 5
DAY_IN_MINUTES = 390


def build_state_vector(o, c, h, l, v, avg_price, idx, debug=False):
    """ idx - is the candle index in range 0 ... 390 """
    padding_size = DAY_IN_MINUTES - (idx + 1)
    padding = [0] * padding_size

    if debug:
        print("calc_normalized_state")
        print('len(o)', len(o))
        print('len(v)', len(v))
        print('padding_size:', padding_size)

    o = np.concatenate((padding, o))
    c = np.concatenate((padding, c))
    h = np.concatenate((padding, h))
    l = np.concatenate((padding, l))
    v = np.concatenate((padding, v))

    if debug:
        print('padded len(o)', len(o))
        print('padded len(v)', len(v))

    if avg_price is None:
        avg_price = 0
        position_size = 0
    else:
        position_size = 1

    state = np.concatenate((o, c, h, l, v, [avg_price], [position_size]))

    return state


ZERO_VEC = [0] * DAY_IN_MINUTES


class Trade_Env:
    def __init__(self, movers, simulation_mode, sim_chart_index=None):
        self._sim_chart_index = sim_chart_index

        self.movers = movers
        self.num_movers = len(movers)

        self.buy_price = None
        self.buy_price_normalized = None

        self.num_trades = 0

        self.df = None
        self.symbol = None
        self.date = None

        self.idx = 0

        self._open_n = None
        self._close_n = None
        self._high_n = None
        self._low_n = None
        self._volume_n = None

        self._open = None
        self._close = None
        self._high = None
        self._low = None
        self._volume = None

        self._time = None   # used for debugging only

        self._state = None
        self._state_normalized = None
        self._render = False

        self.entries_normalized = []
        self.exits_normalized = []

        self.entries = []
        self.exits = []

        self.simulation_mode = simulation_mode

        self.reset()

    def set_sim_chart_idx(self, sim_chart_index):
        self._sim_chart_index = sim_chart_index

    def reset(self, debug=False):
        self._pick_chart()

        o = self.df.Open.to_list()
        c = self.df.Close.to_list()
        h = self.df.High.to_list()
        l = self.df.Low.to_list()
        v = self.df.Volume.to_list()

        self._open = o
        self._close = c
        self._high = h
        self._low = l
        self._volume = v

        price = np.concatenate((o, c, h, l))
        price = cu.normalize_middle(price)
        price = price.reshape(4, DAY_IN_MINUTES)

        o = price[0]
        c = price[1]
        h = price[2]
        l = price[3]
        v = cu.normalize_middle(np.array(v))

        self._open_n = o
        self._close_n = c
        self._high_n = h
        self._low_n = l
        self._volume_n = v

        self._time = self.df.Time.to_list()

        self.idx = 0

        self.entries_normalized = []
        self.exits_normalized = []
        self.entries = []
        self.exits = []

        self.buy_price = None
        self.buy_price_normalized = None

        self.num_trades = 0

        if debug:
            print("RESET", self.idx, self._time[-1], self._time[self.idx])

        self._calc_states(debug)

        return self._state_normalized

    def _calc_states(self, debug):
        _time = self._time[:self.idx]

        o = self._open_n[:self.idx + 1]
        c = self._close_n[:self.idx + 1]
        h = self._high_n[:self.idx + 1]
        l = self._low_n[:self.idx + 1]
        v = self._volume_n[:self.idx + 1]

        if debug:
            print("calc_state len(o)", len(o), "idx:", self.idx)

        s = build_state_vector(o, c, h, l, v, self.buy_price, self.idx, debug)

        self._state_normalized = torch.tensor(s, dtype=torch.float).unsqueeze(0).unsqueeze(0).to("cuda")

        o = self._open[:self.idx + 1]
        c = self._close[:self.idx + 1]
        h = self._high[:self.idx + 1]
        l = self._low[:self.idx + 1]
        v = self._volume[:self.idx + 1]

        if debug:
            print("calc_state len(o)", len(o), "idx:", self.idx)

        self._state = build_state_vector(o, c, h, l, v, self.buy_price, self.idx, debug)

        # print(self.idx, self.close_idx, self._time[-1], self._time[self.idx], self.buy_locations_vector)
        # print(self.idx, self.buy_locations_vector, self.buy_indexes)

    def step(self, action, debug=False):
        if debug:
            print("STEP", self.idx, self._time[-1], self._time[self.idx])

        done = False
        reward = 0
        gain = 0
        trade_done = False

        if self.idx == DAY_IN_MINUTES - 1:
            if self.buy_price_normalized is not None:
                trade_done = True

                self.num_trades += 1

                buy_price_normalized = self.buy_price_normalized
                bpn = cu.shift_and_scale([buy_price_normalized])[0]

                sell_price_normalized = self._close_n[self.idx]
                spn = cu.shift_and_scale([sell_price_normalized])[0]

                reward = spn - bpn

                self.buy_price_normalized = None

                if self.simulation_mode:
                    buy_price = self.buy_price
                    sell_price = self._close[self.idx]
                    gain = 100 * (sell_price / buy_price - 1)

                    c = "green" if reward > 0 else "red"

                    print(colored("%s  BUY: %.2f (%.2f)" % (self.symbol, buy_price_normalized, bpn), color=c), end="")
                    print(colored("   SELL %.2f (%.2f)   REWARD: %.2f" % (sell_price_normalized, spn, reward), color=c))

                    self.exits.append([self._time[self.idx], sell_price])
                    self.exits_normalized.append([self._time[self.idx], sell_price_normalized])
                else:
                    print(colored("XXX", color='red'), end="")

            done = True
            print("Trades:", self.num_trades)

        elif self.idx < DAY_IN_MINUTES:
            if action == 0:  # BUY
                if self.buy_price_normalized is None:
                    buy_price_normalized = self._close_n[self.idx]

                    self.buy_price_normalized = buy_price_normalized

                    if self.simulation_mode:
                        buy_price = self._close[self.idx]
                        self.buy_price = buy_price

                        self.entries.append([self._time[self.idx], buy_price])
                        self.entries_normalized.append([self._time[self.idx], buy_price_normalized])
                    else:
                        print(".", end="")
            elif action == 1:  # SELL
                if self.buy_price_normalized is not None:
                    trade_done = True

                    self.num_trades += 1

                    buy_price_normalized = self.buy_price_normalized
                    bpn = cu.shift_and_scale([buy_price_normalized])[0]

                    sell_price_normalized = self._close_n[self.idx]
                    spn = cu.shift_and_scale([sell_price_normalized])[0]

                    reward = spn - bpn

                    self.buy_price_normalized = None

                    if self.simulation_mode:
                        buy_price = self.buy_price
                        sell_price = self._close[self.idx]
                        gain = 100 * (sell_price / buy_price - 1)

                        self.exits.append([self._time[self.idx], sell_price])
                        self.exits_normalized.append([self._time[self.idx], sell_price_normalized])

                        if debug:
                            c = "green" if reward > 0 else "red"
                            print(colored("%s  BUY: %.2f (%.2f)" % (self.symbol, buy_price_normalized, bpn), color=c), end="")
                            print(colored("   SELL %.2f (%.2f)   REWARD: %.2f" % (sell_price_normalized, spn, reward), color=c))
                        else:
                            c = "green" if gain > 0 else "red"
                            print(colored("%s  BUY: %.2f" % (self.symbol, buy_price), color=c), end="")
                            print(colored("   SELL %.2f   GAIN: %.2f" % (sell_price, gain), color=c))
                    else:
                        c = "green" if reward > 0 else "red"
                        print(colored("|", color=c), end="")
                elif action == 2:  # Idle
                    reward = 0

        if self.idx < DAY_IN_MINUTES - 1:
            self.idx += 1

            if not self.simulation_mode:
                print("_", end="")

        self._calc_states(debug)
        return self._state_normalized, reward, gain, trade_done, done

    def _pick_chart(self):
        open_idx = None
        close_idx = None

        while (open_idx is None) or (close_idx is None):
            if self.simulation_mode:
                if self._sim_chart_index is None:
                    rand_idx = random.randint(int(self.num_movers * 0.8) + 1, self.num_movers - 1)
                else:
                    rand_idx = self._sim_chart_index
            else:
                if self._sim_chart_index is None:
                    rand_idx = random.randint(0, int(self.num_movers * 0.8))
                else:
                    rand_idx = self._sim_chart_index

            self.symbol = self.movers.iloc[rand_idx]["symbol"]
            self.date = self.movers.iloc[rand_idx]["date"]

            self.df = cu.get_chart_data_prepared_for_ai(self.symbol, self.date)

            print(self.symbol, self.date)

            if self.df is not None:
                open_idx = cu.get_time_index(self.df, self.date, 9, 30, 0)
                close_idx = cu.get_time_index(self.df, self.date, 15, 59, 0)

    def save_chart(self, filename=None):
        # if True:
        print("Save Chart: ", len(self.entries))
        if len(self.entries) > 0:
            s = self._state
            s = s[:-2]
            s = s.reshape(DATA_ROWS, DAY_IN_MINUTES)

            o = s[0]
            c = s[1]
            h = s[2]
            l = s[3]
            v = cu.shift_and_scale(s[4], bias=0.5, scale_factor=0.5)

            t = self._time[0:self.idx+1]

            dx = pd.DataFrame({
                'Time': t,
                'Open': o,
                'Close': c,
                'High': h,
                'Low': l,
                'Volume': v})

            if filename is None:
                filename = self.symbol
            else:
                filename = self.symbol + "_" + self.date + "_" + filename

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

    def save_normalized_chart(self, filename=None):
        # if True:
        if len(self.entries_normalized) > 0:
            s = self._state_normalized.to("cpu")
            s = s.numpy()
            s = s[0][0]
            s = s.reshape(DATA_ROWS, DAY_IN_MINUTES)

            o = s[0]
            c = s[1]
            h = s[2]
            l = s[3]
            v = cu.shift_and_scale(s[4], bias=0.5, scale_factor=0.5)

            t = self._time[0:self.idx+1]

            dx = pd.DataFrame({
                'Time': t,
                'Open': o,
                'Close': c,
                'High': h,
                'Low': l,
                'Volume': v})

            if filename is None:
                filename = self.symbol
            else:
                filename = self.symbol + "_" + filename

            images_dir_path = "trades\\"
            cu.show_1min_chart_normalized(dx,
                                          self.idx,
                                          self.symbol,
                                          self.date,
                                          "",
                                          self.entries_normalized,
                                          self.exits_normalized,
                                          [],
                                          [],
                                          images_dir_path,
                                          filename)

    @property
    def num_states(self):
        return len(self._state_normalized)

    @property
    def num_actions(self):
        return 3

    @property
    def state(self):
        return self._state_normalized

    @property
    def state_shape(self):
        _, _h, _w = self._state_normalized.shape
        return _h, _w
