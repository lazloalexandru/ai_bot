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

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0001
GAMMA = 0.99
BATCH_SIZE = 50


class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size

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
        fc1 = tf.layers.dense(self._states, 1000, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 1000, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

        self._saver = tf.train.Saver(max_to_keep=5)

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states: state.reshape(1, self.num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})
        self._model_valid = True

    def save(self, sess, step):
        self._saver.save(sess, "checkpoints\\my_model", global_step=step)

    def restore(self, sess):
        path = tf.train.latest_checkpoint('checkpoints\\')
        print("Checkpoint: ", path)
        self._saver.restore(sess, path)

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return self._num_actions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def var_init(self):
        return self._var_init


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)


class TradeBot:
    def __init__(self, sess, model, env, memory, max_eps, min_eps, decay, render=True):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._eps = self._max_eps
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

            self._memory.add_sample((state, action, reward, next_state))

            # exponentially decay the eps value
            self._steps += 1
            self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                self._env.save_traded_chart()
                self._reward_store.append(tot_reward)
                break

        print("Step {}, Account: {}, Eps: {}".format(self._steps, tot_reward, self._eps))

    def _choose_action(self, state):
        return np.argmax(self._model.predict_one(state, self._sess))

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def max_x_store(self):
        return self._max_x_store


DAY_IN_MINUTES = 720


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
        self._time = None   # used for debugging only

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
        _price = tf.keras.utils.normalize(_price)

        _open = np.concatenate((padding, _price[0].reshape(n)))
        _close = np.concatenate((padding, _price[1].reshape(n)))
        _high = np.concatenate((padding, _price[2].reshape(n)))
        _low = np.concatenate((padding, _price[3].reshape(n)))

        _volume = self._volume[:self.idx + 1]
        _volume = tf.keras.utils.normalize(_volume).reshape(n)
        _volume = np.concatenate((padding, _volume), axis=None)

        self._state = np.concatenate(([self.position_size], _open, _close, _high, _low, _volume))

        return self._state

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

        self._calc_state()

        return self._state

    def step(self, action):
        self.idx += 1
        self._calc_state()

        done = False
        gain = 0

        if self.idx == self.df.index[-1]:
            done = True
            if self.position_size == 1:
                self.position_size = 0
                sell_price = self._close[-1]
                gain = 100 * (sell_price / self.entry_price - 1)
                self.exits.append([self._time[self.idx], sell_price])

                print(colored("   SELL %s   GAIN: %s" % (sell_price, gain), color="green" if gain > 0 else "red"))

            print(colored("Trades: %s" % len(self.entries), color="green"))

        else:
            if action == 0:  # BUY
                if self.position_size == 0:
                    self.position_size = 1
                    self.entry_price = self._close[self.idx]
                    self.entries.append([self._time[self.idx], self.entry_price])
                    print(self.symbol, 'BUY:', self.entry_price, end="")

            elif action == 1:  # Idle
                print("___", end="")
                gain = 0
            elif action == 2:  # SELL
                if self.position_size == 1:
                    sell_price = self._close[self.idx]
                    gain = 100 * (sell_price / self.entry_price - 1)
                    self.position_size = 0
                    self.exits.append([self._time[self.idx], sell_price])

                    print(colored("   SELL %s   GAIN: %s" % (sell_price, gain), color="green" if gain > 0 else "red"))

        return self._state, gain, done

    def _pick_chart(self):
        open_idx = None
        close_idx = None

        while (open_idx is None) or (close_idx is None):
            rand_idx = random.randint(0, self.num_movers - 1)
            rand_idx = 2322
            self.df = cu.get_chart_data_prepared_for_ai(self.movers.iloc[rand_idx])

            self.symbol = self.movers.iloc[rand_idx]["symbol"]
            self.date = self.movers.iloc[rand_idx]["date"]

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
                               images_dir_path)

    @property
    def num_states(self):
        return len(self._state)

    @property
    def num_actions(self):
        return 3

    @property
    def get_state(self):
        return self._state


def test():
    if not os.path.isfile(__active_days_file):
        print(colored("ERROR: " + __active_days_file + " not found!", color="red"))
    else:
        movers = pd.read_csv(__active_days_file)

        tr = Trade_Env(movers)

        model = Model(tr.num_states, tr.num_actions, BATCH_SIZE)
        mem = Memory(50000)

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(model.var_init)
            model.restore(sess)
            bot = TradeBot(sess, model, tr, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA, False)
            num_episodes = 100
            cnt = 0
            while cnt < num_episodes:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
                bot.run()
                cnt += 1

            plt.plot(bot.reward_store)
            plt.show()
            plt.close("all")


if __name__ == "__main__":
    test()
