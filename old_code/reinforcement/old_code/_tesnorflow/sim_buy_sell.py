import os
import mplfinance as mpf
import numpy as np
import tensorflow.compat.v1 as tf
from termcolor import colored
import common as cu
tf.compat.v1.disable_eager_execution()
import matplotlib.pylab as plt
import random
import math
import pandas as pd

__active_days_file = "data\\active_days.csv"


class Model:
    def __init__(self, num_states, num_actions):
        self._num_states = num_states
        self._num_actions = num_actions

        self._model_valid = False

        # define the placeholders
        self._states = None
        self._actions = None

        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None

        self._saver = None

        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 10000, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 5000, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 5000, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc3, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

        self._saver = tf.train.Saver(max_to_keep=5)

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states: state.reshape(1, self.num_states)})

    def save(self, sess, step):
        self._saver.save(sess, "checkpoints\\my_model", global_step=step)

    def restore(self, sess):
        res = False

        path = tf.train.latest_checkpoint('checkpoints\\')
        print("Checkpoint: ", path)
        if path is None:
            sess.run(self.var_init)
        else:
            self._saver.restore(sess, path)
            res = True

        return res

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def var_init(self):
        return self._var_init


class TradeBot:
    def __init__(self, sess, model, env, render=True):
        self._sess = sess
        self._env = env
        self._model = model
        self._render = render
        self._steps = 0
        self._reward_store = []
        self._max_x_store = []

    def run(self):
        state = self._env.reset()
        tot_reward = 0

        while True:
            action = self._choose_action(state)
            next_state, reward, done = self._env.step(action)

            if done:
                next_state = None

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                c = 'red' if tot_reward < 0 else 'green'
                print(colored("Account: %.2f" % tot_reward, color=c), '\n')
                self._env.save_normalized_chart()
                self._reward_store.append(tot_reward)
                break

    def _choose_action(self, state):
        return np.argmax(self._model.predict_one(state, self._sess))

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def max_x_store(self):
        return self._max_x_store


DAY_IN_MINUTES = 390


class Trade_Env:
    def __init__(self, movers):
        self.movers = movers
        self.num_movers = len(movers)

        self.position_size = 0
        self.entry_price = 0

        self.df = None
        self.symbol = None
        self.date = None
        self.idx = None

        self._open = None
        self._close = None
        self._high = None
        self._low = None
        self._volume = None
        self._time = None   # used for debugging only!

        self._state = None
        self._render = False

        self.entries = []
        self.exits = []

        self.reset()

    def _calc_state(self):
        _time = self._time[:self.idx + 1]

        n = len(_time)

        padding_size = DAY_IN_MINUTES - len(_time)
        padding = [0] * padding_size

        _open = self._open[:self.idx + 1]
        _close = self._close[:self.idx + 1]
        _high = self._high[:self.idx + 1]
        _low = self._low[:self.idx + 1]

        _price = [_open, _close, _high, _low]
        _price = tf.keras.utils.normalize_middle(_price)

        _open = np.concatenate((padding, _price[0].reshape(n)))
        _close = np.concatenate((padding, _price[1].reshape(n)))
        _high = np.concatenate((padding, _price[2].reshape(n)))
        _low = np.concatenate((padding, _price[3].reshape(n)))

        _volume = self._volume[:self.idx + 1]
        _volume = tf.keras.utils.normalize_middle(_volume).reshape(n)
        _volume = np.concatenate((padding, _volume), axis=None)

        xxx_time = (100 * _time[self.idx].time().hour + _time[self.idx].time().minute) / 1600

        self._state = np.concatenate(([self.position_size], [xxx_time], _open, _close, _high, _low, _volume))

        return self._state

    def reset(self):
        self._pick_chart()
        self._open = self.df.Open.to_list()
        self._close = self.df.Close.to_list()
        self._high = self.df.High.to_list()
        self._low = self.df.Low.to_list()
        self._volume = self.df.Volume.to_list()
        self._time = self.df.Time.to_list()
        self.position_size = 0
        self.entries = []
        self.exits = []

        self._calc_state()

        return self._state

    def step(self, action):
        self.idx += 1
        self._calc_state()

        FEES = 1  # in %
        done = False
        gain = 0

        if self.idx == self.df.index[-1]:
            if self.position_size == 1:
                sell_price = self._close[self.idx]
                gain = 100 * (sell_price / self.entry_price - 1) - FEES
                self.exits.append([self._time[self.idx], sell_price])
                print(colored("   SELL %s   GAIN: %.2f" % (sell_price, gain), color="green" if gain > 0 else "red"))
                print("----------------------------------")

            self.position_size = 0
            done = True
            print("Trades: %s" % len(self.entries))
        else:
            if action == 0:  # BUY
                if self.position_size == 0:
                    self.position_size = 1
                    self.entry_price = self._close[self.idx]
                    self.entries.append([self._time[self.idx], self.entry_price])
                    print(self.symbol, 'BUY:', self.entry_price, end="")
            elif action == 1:  # SELL
                if self.position_size == 1:
                    sell_price = self._close[self.idx]
                    gain = 100 * (sell_price / self.entry_price - 1)
                    self.position_size = 0
                    self.exits.append([self._time[self.idx], sell_price])

                    print(colored("   SELL %s   GAIN: %.2f" % (sell_price, gain), color="green" if gain > 0 else "red"))

        return self._state, gain, done

    def _pick_chart(self):
        open_idx = None
        close_idx = None

        while (open_idx is None) or (close_idx is None):
            rand_idx = random.randint(int(self.num_movers * 0.8) + 1, self.num_movers - 1)
            rand_idx = 2300
            self.symbol = self.movers.iloc[rand_idx]["symbol"]
            self.date = self.movers.iloc[rand_idx]["date"]

            self.df = cu.get_chart_data_prepared_for_ai(self.symbol, self.date)

            print(self.symbol, self.date)

            if self.df is not None:
                open_idx = cu.get_time_index(self.df, self.date, 9, 30, 0)
                close_idx = cu.get_time_index(self.df, self.date, 15, 59, 0)

        self.idx = open_idx

    def save_traded_chart(self):
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
                               images_dir_path)

    @property
    def num_states(self):
        return len(self._state)

    @property
    def num_actions(self):
        return 2

    @property
    def get_state(self):
        return self._state


def stats(gains):
    plus = sum(x > 0 for x in gains)
    splus = sum(x for x in gains if x > 0)
    minus = sum(x < 0 for x in gains)
    sminus = sum(x for x in gains if x < 0)

    num_trades = len(gains)
    success_rate = None if (plus + minus) == 0 else round(100 * (plus / (plus + minus)))
    rr = None if plus == 0 or minus == 0 else -(splus / plus) / (sminus / minus)

    avg_win = None if plus == 0 else splus / plus
    avg_loss = None if minus == 0 else sminus / minus

    if len(gains) > 0:
        print("")
        print("Nr Trades:", num_trades)
        print("Success Rate:", success_rate, "%")
        print("R/R:", "N/A" if rr is None else "%.2f" % rr)
        print("Winners:", plus, " Avg. Win:", "N/A" if avg_win is None else "%.2f" % avg_win + "%")
        print("Losers:", minus, " Avg. Loss:", "N/A" if avg_loss is None else "%.2f" % avg_loss + "%")
        print("")

    x = list(range(0, len(gains)))
    plt.bar(x, gains)
    plt.show()
    plt.close("all")


def test():
    if not os.path.isfile(__active_days_file):
        print(colored("ERROR: " + __active_days_file + " not found!", color="red"))
    else:
        movers = pd.read_csv(__active_days_file)

        tr = Trade_Env(movers)

        model = Model(tr.num_states, tr.num_actions)

        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:
            if model.restore(sess):
                bot = TradeBot(sess, model, tr, False)
                num_episodes = 10
                cnt = 0
                while cnt < num_episodes:
                    print('Episode {} of {}'.format(cnt+1, num_episodes))
                    bot.run()
                    cnt += 1

                stats(bot.reward_store)


if __name__ == "__main__":
    test()
