import os
import numpy as np
import tensorflow.compat.v1 as tf
from termcolor import colored
import common as cu
import matplotlib.pylab as plt
import random
import math
import pandas as pd

tf.compat.v1.disable_eager_execution()

__active_days_file = "data\\active_days.csv"

MAX_EPSILON = 1.0
MIN_EPSILON = 0.1
LAMBDA = 0.0001
GAMMA = 0.99

EPISODES = 100
SAVE_EPISODE_STEP = 50

MEMORY_SIZE = 1000
BATCH_SIZE = 100

TRAINING_START = 390
TRAINING_STEP = 20


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

        # now setup the model
        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 1000, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 500, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 500, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc3, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

        self._saver = tf.train.Saver(max_to_keep=100)

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
        if path is None:
            sess.run(self.var_init)
        else:
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


class TrainerBot:
    def __init__(self, sess, model, env, memory, render=True):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._render = render
        self._eps = MAX_EPSILON
        self._steps = 0
        self._reward_store = []
        self._max_x_store = []

    def run(self):
        state = self._env.reset()
        tot_reward = 0

        if self._steps > TRAINING_START:
            print("Training Active. Step:", self._steps)
        else:
            print("Random Simulation. Step:", self._steps)

        while True:
            action = self._choose_action(state)
            next_state, reward, done = self._env.step(action)

            if done:
                next_state = None

            self._memory.add_sample((state, action, reward, next_state))
            if self._steps > TRAINING_START:
                if self._steps % TRAINING_STEP == 0:
                    self._replay()

            # exponentially decay the eps value
            self._steps += 1
            self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                self._reward_store.append(tot_reward)
                break

        c = 'red' if tot_reward < 0 else 'green'
        print("Account: ", colored("%.2f" % tot_reward, color=c), "Eps:", self._eps)
        '''
        gx = 'X' * int(abs(tot_reward))
        print(colored(gx, color=c))
        print("\n")
        '''

    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model.num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))

        return np.argmax(self._model.predict_one(state, self._sess))

    def _replay(self):
        batch = self._memory.sample(self._model.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model.num_states) if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model.num_states))
        y = np.zeros((len(batch), self._model.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def max_x_store(self):
        return self._max_x_store


class Trade_Env:
    def __init__(self, movers):
        self.movers = movers
        self.num_movers = len(movers)
        self.entry_price = 0
        self.entries = 0
        self.df = None
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

        self.reset()

    def _calc_state(self):
        _time = self._time[:self.idx + 1]

        o = self._open[:self.idx + 1]
        c = self._close[:self.idx + 1]
        h = self._high[:self.idx + 1]
        l = self._low[:self.idx + 1]
        v = self._volume[:self.idx + 1]

        state = cu.calc_normalized_state(o, c, h, l, v, self.idx - self.open_idx)

        self._state = state

    def reset(self):
        self._pick_chart()
        self._open = self.df.Open.to_list()
        self._close = self.df.Close.to_list()
        self._high = self.df.High.to_list()
        self._low = self.df.Low.to_list()
        self._volume = self.df.Volume.to_list()
        self._time = self.df.Time.to_list()

        self.entries = 0
        self._calc_state()

        return self._state

    def calc_reward_buy(self):
        reward = 0
        risk = -1

        stop = False
        i = self.idx + 1
        while not stop and i < self.close_idx:
            mn = 100 * (self._low[i] / self.entry_price - 1)
            mx = 100 * (self._high[i] / self.entry_price - 1)

            if mn < risk:
                risk = mn

            if mx > reward:
                reward = mx
            i += 1

        return (reward / abs(risk)) - 2

    def step(self, action):
        self.idx += 1
        self._calc_state()

        # print(self.idx, self.close_idx, self._time[-1], self._time[self.idx])

        FEES = 1
        done = False
        reward = 0

        if self.idx >= self.close_idx:
            done = True
            print("Entries:", self.entries)
        else:
            if action == 0:  # BUY
                self.entries += 1
                self.entry_price = self._close[self.idx]
                reward = self.calc_reward_buy()

            elif action == 1:  # SELL
                reward = 0

        return self._state, reward, done

    def _pick_chart(self):
        open_idx = None
        close_idx = None

        while (open_idx is None) or (close_idx is None):
            rand_idx = random.randint(0, int(self.num_movers * 0.8))
            rand_idx = 2300
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

    @property
    def num_states(self):
        return len(self._state)

    @property
    def num_actions(self):
        return 2

    @property
    def get_state(self):
        return self._state


def train():
    if not os.path.isfile(__active_days_file):
        print(colored("ERROR: " + __active_days_file + " not found!", color="red"))
    else:
        movers = pd.read_csv(__active_days_file)

        tr = Trade_Env(movers)

        print(tr.num_actions, tr.num_states)

        model = Model(tr.num_states, tr.num_actions, BATCH_SIZE)
        mem = Memory(MEMORY_SIZE)

        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:
            model.restore(sess)

            bot = TrainerBot(sess, model, tr, mem, False)
            num_episodes = EPISODES
            cnt = 0
            while cnt < num_episodes:
                print('\nEpisode {} of {}'.format(cnt+1, num_episodes))
                bot.run()
                cnt += 1
                if cnt % SAVE_EPISODE_STEP == 0:
                    model.save(sess, cnt)

            plt.plot(bot.reward_store)
            plt.show()
            plt.close("all")


if __name__ == "__main__":
    # test()
    train()
