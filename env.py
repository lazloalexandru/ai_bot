import numpy as np
from termcolor import colored
import common as cu
import random


class Trade_Env:
    def __init__(self, movers, sim_chart_index=None, simulation_mode=True):

        self.sim_chart_index = sim_chart_index

        self.movers = movers
        self.num_movers = len(movers)

        self.buy_prices = []
        self.num_trades = 0
        self.buy_locations_vector = [0] * cu.DAY_IN_MINUTES
        self.buy_indexes = []

        self.df = None
        self.symbol = None
        self.date = None
        self.idx = None
        self.open_idx = None
        self.close_idx = None

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

    def _calc_state(self):
        _time = self._time[:self.idx + 1]

        o = self._open[:self.idx + 1]
        c = self._close[:self.idx + 1]
        h = self._high[:self.idx + 1]
        l = self._low[:self.idx + 1]
        v = self._volume[:self.idx + 1]

        state = cu.calc_normalized_state(o, c, h, l, v, self.idx - self.open_idx)
        self._state = np.concatenate((self.buy_locations_vector, state))

    def reset(self):
        self._pick_chart()
        self._open = self.df.Open.to_list()
        self._close = self.df.Close.to_list()
        self._high = self.df.High.to_list()
        self._low = self.df.Low.to_list()
        self._volume = self.df.Volume.to_list()
        self._time = self.df.Time.to_list()

        self.entries = []
        self.exits = []

        self.buy_locations_vector = [0] * cu.DAY_IN_MINUTES
        self.buy_indexes = []
        self.buy_prices = []
        self.num_trades = 0

        self._calc_state()

        return self._state

    def step(self, action):
        self.idx += 1

        # print(self.idx, self.close_idx, self._time[-1], self._time[self.idx])

        FEES = 1  # in %
        done = False
        gain = 0

        num_positions = len(self.buy_prices)
        if self.idx >= self.close_idx:
            if num_positions > 0:
                self.num_trades += num_positions

                self.buy_locations_vector[self.idx] = [0] * cu.DAY_IN_MINUTES

                avg_price = sum(self.buy_prices) / num_positions
                sell_price = self._close[self.idx]
                gain = 100 * (sell_price / avg_price - 1) - FEES

                if not self.simulation_mode:
                    gain -= 50

                gain = gain * num_positions

                if self.simulation_mode:
                    self.exits.append([self._time[self.idx], sell_price])
                    print(self.symbol, 'BUY:', avg_price, end="")
                    print(colored("   SELL %.2f  x  %s   GAIN: %.2f" % (sell_price, num_positions, gain), color="green" if gain > 0 else "red"))
                else:
                    print(colored("XXX", color='red'), end="")

            done = True
            print("  Trades:", self.num_trades)
        else:
            if action == 0:  # BUY
                self.buy_prices.append(self._close[self.idx])
                self.buy_indexes.append(self.idx)
                self.buy_locations_vector[self.idx] = 1

                if self.simulation_mode:
                    self.entries.append([self._time[self.idx], self._close[self.idx]])
                else:
                    print(".", end="")
            elif action == 1:  # SELL
                if num_positions > 0:
                    self.num_trades += 1

                    entry_price = self.buy_prices.pop()
                    buy_index = self.buy_indexes.pop()
                    self.buy_locations_vector[buy_index] = 0

                    sell_price = self._close[self.idx]
                    gain = 100 * (sell_price / entry_price - 1) - FEES

                    if self.simulation_mode:
                        self.exits.append([self._time[self.idx], sell_price])

                        print(self.symbol, 'BUY:', entry_price, end="")
                        c = "green" if gain > 0 else "red"
                        print(colored("   SELL %.2f   GAIN: %.2f" % (sell_price, gain), color=c))
                    else:
                        print("|", end="")

            elif action == 2:  # IDLE
                gain = 0

        self._calc_state()
        return self._state, gain, done

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

        self.open_idx = open_idx
        self.close_idx = close_idx

        self.idx = open_idx

    def save_traded_chart(self, filename=None):
        if len(self.entries) > 0:

            images_dir_path = "trades\\"
            cu.show_1min_chart(self.df,
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
    def get_state(self):
        return self._state
