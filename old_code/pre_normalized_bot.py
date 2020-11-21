import datetime
import os

import numpy as np
import tensorflow.compat.v1 as tf
from termcolor import colored
import common as cu
tf.compat.v1.disable_eager_execution()
import matplotlib.pylab as plt
import random
import math
import pandas as pd


MAX_EPSILON = 1.0
MIN_EPSILON = 0.9999
LAMBDA = 0.00001
GAMMA = 0.99
BATCH_SIZE = 500


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
        fc1 = tf.layers.dense(self._states, 10000, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 5000, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 5000, activation=tf.nn.relu)
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
    def __init__(self, sess, model, env, memory):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._eps = MAX_EPSILON
        self._steps = 0
        self._cnt = 0
        self._reward_store = []
        self._max_x_store = []

    def run(self):
        state = self._env.reset()
        tot_reward = 0

        if self._steps > 20000:
            print("Training Active. Step:", self._steps)
        else:
            print("Random Simulation. Step:", self._steps)

        while True:
            action = self._choose_action(state)
            next_state, reward, done = self._env.step(action)

            if done:
                next_state = None

            self._memory.add_sample((state, action, reward, next_state))
            if self._steps > 20000:
                if self._steps % 100 == 0:
                    self._replay()

            if self._steps % 10 == 0:
                self._cnt += 1

            # exponentially decay the eps value
            self._steps += 1
            self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self._cnt)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                self._reward_store.append(tot_reward)
                break

        gx = 'X' * int(abs(tot_reward))
        c = 'red' if tot_reward < 0 else 'green'
        print("Account: ", colored("%.2f" % tot_reward, color=c), "Eps:", self._eps)
        print(colored(gx, color=c))
        print("\n")

    def _choose_action(self, state):
        if random.random() < self._eps:
            action = random.randint(0, self._model.num_actions - 1)
        else:
            action = np.argmax(self._model.predict_one(state, self._sess))

        return action

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

        self._entries = 0
        self._position_size = 0
        self.entry_price = 0

        self.idx = 0

        self._chart_states = None

        self.reset()

    def reset(self):
        self._entries = 0
        self._position_size = 0
        self.entry_price = 0

        self.idx = 0

        self._chart_states = None

        self._pick_chart()

        print(self._chart_states[0].shape)
        return np.concatenate(([self._position_size], self._chart_states[0]))

    def get_close_price(self):
        state = self._chart_states[self.idx]
        state = state[:-1]
        state = state.reshape(5, cu.DAY_IN_MINUTES)
        c = state[1]

        return c[-1]

    def calc_gain(self):
        sell_price = self.get_close_price()
        if self.entry_price > 0:
            gain = 100 * (sell_price - self.entry_price) / self.entry_price - FEES

    def step(self, action):
        self.idx += 1

        FEES = 1
        done = False
        gain = 0

        if self.idx >= cu.DAY_IN_MINUTES - 1:
            if self._position_size == 1:
                print(colored("XXX", color='red'), end="")

                sell_price = self.get_close_price()

                gain = 100 * (sell_price / self.entry_price - 1) - FEES
                self._position_size = 0

            done = True

            print("   Entries:", self._entries)
        else:
            if action == 0:  # BUY
                if self._position_size == 0:
                    self._entries += 1
                    self._position_size = 1
                    self.entry_price = self.get_close_price()
                    gain = 0
                    print("-", end="")

            elif action == 1:  # SELL
                if self._position_size == 1:
                    sell_price = self.get_close_price()
                    gain = 100 * (sell_price / self.entry_price - 1) - FEES
                    self._position_size = 0
                    print("|", end="")

        state = np.concatenate(([self._position_size], self._chart_states[self.idx]))
        return state, gain, done

    def _pick_chart(self):
        n = len(self.movers)
        if n > 0:
            num_tries = 0

            while num_tries < n and self._chart_states is None:
                num_tries += 1  # needed to avoid infinite loop if files are missing

                rand_idx = random.randint(0, int(self.num_movers * 0.8))

                symbol = self.movers.iloc[rand_idx]['symbol']
                date_ = self.movers.iloc[rand_idx]['date']
                self._chart_states = cu.get_normalized_states(symbol, date_)

                if self._chart_states is not None:
                    self.symbol = self.movers.iloc[rand_idx]["symbol"]
                    self.date = self.movers.iloc[rand_idx]["date"]
                    print(self.symbol, self.date)

    @property
    def num_states(self):
        res = None

        if self._chart_states is not None:
            res = len(self._chart_states[0])

        return res

    @property
    def num_actions(self):
        return 2

    @property
    def get_state(self):
        return self._state


def train():
    if not os.path.isfile(cu.__active_days_file):
        print(colored("ERROR: " + cu.__active_days_file + " not found!", color="red"))
    else:
        movers = pd.read_csv(cu.__active_days_file)

        tr = Trade_Env(movers)

        print(tr.num_actions, tr.num_states)

        model = Model(tr.num_states, tr.num_actions, BATCH_SIZE)
        mem = Memory(50000)

        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:
            model.restore(sess)

            bot = TrainerBot(sess, model, tr, mem)
            num_episodes = 10000
            cnt = 0
            while cnt < num_episodes:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
                bot.run()
                cnt += 1
                if cnt % 500 == 0:
                    model.save(sess, cnt)

            plt.plot(bot.reward_store)
            plt.show()
            plt.close("all")


if __name__ == "__main__":
    train()
