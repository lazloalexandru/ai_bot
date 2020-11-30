import numpy as np
from termcolor import colored
import common as cu
import random
import torch
import pandas as pd

DATA_ROWS = 7
DAY_IN_MINUTES = 390


def build_state_vector(o, c, h, l, v, b, idx, debug=False):
    """ idx - is the candle index in range 0 ... 390 """
    padding_size = DAY_IN_MINUTES - (idx + 1)
    padding = [0] * padding_size

    if debug:
        print("calc_normalized_state")
        print('len(o)', len(o))
        print('len(v)', len(v))
        print('len(b)', len(b))
        print('padding_size:', padding_size)

    o = np.concatenate((padding, o))
    c = np.concatenate((padding, c))
    h = np.concatenate((padding, h))
    l = np.concatenate((padding, l))
    v = np.concatenate((padding, v))
    t = [idx / DAY_IN_MINUTES]

    if debug:
        print('padded len(o)', len(o))
        print('padded len(v)', len(v))
        print('padded len(b)', len(b))

    state = np.concatenate((o, c, h, l, v, b, t))

    return state


ZERO_VEC = [0] * DAY_IN_MINUTES


class Trade_Env:
    def __init__(self, movers, simulation_mode, sim_chart_index=None):
        self.sim_chart_index = sim_chart_index

        self.movers = movers
        self.num_movers = len(movers)

        self.buy_price = None
        self.buy_locations_vector = ZERO_VEC

        self.num_trades = 0

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

        self._state = None
        self._render = False

        self.entries = []
        self.exits = []

        self.simulation_mode = simulation_mode

        self.reset()

    def reset(self, debug=False):
        self._pick_chart()

        o = self.df.Open.to_list()
        c = self.df.Close.to_list()
        h = self.df.High.to_list()
        l = self.df.Low.to_list()
        v = self.df.Volume.to_list()

        price = np.concatenate((o, c, h, l))
        price = cu.normalize(price)
        price = price.reshape(4, DAY_IN_MINUTES)

        o = price[0]
        c = price[1]
        h = price[2]
        l = price[3]
        v = cu.normalize(np.array(v))

        self._open = o
        self._close = c
        self._high = h
        self._low = l
        self._volume = v
        self._time = self.df.Time.to_list()

        self.idx = 0

        self.entries = []
        self.exits = []

        self.buy_locations_vector = ZERO_VEC
        self.buy_price = None

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

        s = build_state_vector(o, c, h, l, v, self.buy_locations_vector, self.idx, debug)

        self._state = torch.tensor(s, dtype=torch.float).unsqueeze(0).unsqueeze(0).to("cuda")

        # print(self.idx, self.close_idx, self._time[-1], self._time[self.idx], self.buy_locations_vector)
        # print(self.idx, self.buy_locations_vector, self.buy_indexes)

    def step(self, action, debug=False):
        if debug:
            print("STEP", self.idx, self._time[-1], self._time[self.idx], len(self.buy_locations_vector))

        done = False
        reward = 0

        price_offset = 2

        if self.idx == DAY_IN_MINUTES - 1:
            if self.buy_price is not None:
                self.num_trades += 1

                buy_price = self.buy_price + price_offset
                bp = 10 * buy_price

                sell_price = self._close[self.idx]
                sp = 10*(sell_price + price_offset)

                reward = sp - bp

                # Close all positions
                self.buy_locations_vector = ZERO_VEC
                self.buy_price = None

                if self.simulation_mode:
                    c = "green" if reward > 0 else "red"
                    self.exits.append([self._time[self.idx], sell_price])
                    print(colored("%s  BUY: %.2f (%.2f)" % (self.symbol, buy_price, bp), color=c), end="")
                    print(colored("   SELL %.2f (%.2f)   REWARD: %.2f" % (sell_price, sp, reward), color=c))
                else:
                    print(colored("XXX", color='red'), end="")

            done = True
            print("Trades:", self.num_trades)

        elif self.idx < DAY_IN_MINUTES:
            if action == 0:  # BUY
                if self.buy_price is None:
                    buy_price = self._close[self.idx]
                    self.buy_price = buy_price
                    self.buy_locations_vector[self.idx] = 1

                    if self.simulation_mode:
                        self.entries.append([self._time[self.idx], buy_price])
                    else:
                        print(".", end="")
            elif action == 1:  # SELL
                if self.buy_price is not None:
                    self.num_trades += 1

                    buy_price = self.buy_price
                    sell_price = self._close[self.idx]

                    bp = 10 * (buy_price + price_offset)
                    sp = 10 * (sell_price + price_offset)
                    reward = sp - bp

                    self.buy_price = None
                    self.buy_locations_vector = ZERO_VEC

                    if self.simulation_mode:
                        self.exits.append([self._time[self.idx], sell_price])

                        c = "green" if reward > 0 else "red"
                        print(colored("%s  BUY: %.2f (%.2f)" % (self.symbol, buy_price, bp), color=c), end="")
                        print(colored("   SELL %.2f (%.2f)   REWARD: %.2f" % (sell_price, sp, reward), color=c))
                    else:
                        c = "green" if reward > 0 else "red"
                        print(colored("|", color=c), end="")
                elif action == 2:  # Idle
                    reward = 0

        if self.idx < DAY_IN_MINUTES - 1:
            self.idx += 1

            if not self.simulation_mode:
                print("_", end="")

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
        # if True:
        if len(self.entries) > 0:
            # print("save_chart -> len(time)", len(self._time), len(self.entries))

            s = self._state.to("cpu")
            s = s.numpy()
            s = s[0][0]
            s = s[:-1]
            s = s.reshape(6, DAY_IN_MINUTES)

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
            # print("save_chart   ->  idx:", self.idx, len(o), len(t))

            dx = pd.DataFrame({
                'Time': t,
                'Open': o,
                'Close': c,
                'High': h,
                'Low': l,
                'Volume': v})

            filename = self.symbol + "_" + filename
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
