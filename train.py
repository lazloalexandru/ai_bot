import os
import numpy as np
import tensorflow.compat.v1 as tf
from termcolor import colored
import matplotlib.pylab as plt
import random
import math
import pandas as pd
from model import Model
from ai_memory import Memory
from env import Trade_Env

tf.compat.v1.disable_eager_execution()


class TrainerBot:
    def __init__(self, sess, model, env, memory, params, render=True):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory

        self._params = params

        self._eps = self._params['MAX_EPSILON']
        self._MAX_EPSILON = self._params['MAX_EPSILON']
        self._MIN_EPSILON = self._params['MIN_EPSILON']
        self._LAMBDA = self._params['LAMBDA']
        self._GAMMA = self._params['GAMMA']
        self._TRAINING_START = self._params['TRAINING_START']

        self._steps = 0
        self._reward_store = []
        self._performance_store = []

        self._render = render

    def run(self):
        state = self._env.reset()
        tot_reward = 0

        if self._steps > self._TRAINING_START:
            print("Training Active. Step:", self._steps, "Eps:", self._eps)
        else:
            print("Random Simulation. Step:", self._steps, "Eps:", self._eps)

        done = False
        while not done:
            action = self._choose_action(state)
            next_state, reward, done = self._env.step(action)

            if done:
                next_state = None

            self._memory.add_sample((state, action, reward, next_state))

            self._steps += 1

            self._eps = self._MIN_EPSILON + (self._MAX_EPSILON - self._MIN_EPSILON) * math.exp(-self._LAMBDA * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

        if self._steps > self._TRAINING_START:
            self._replay()

        self._reward_store.append(tot_reward)
        gx = '*' * int(abs(tot_reward))
        c = 'red' if tot_reward < 0 else 'green'
        print("Account: ", colored("%.2f" % tot_reward, color=c))
        print(colored(gx, color=c))
        print("\n")

        return tot_reward

    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model.num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))

    def _replay(self):
        batch = self._memory.sample(self._model.batch_size)

        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model.num_states) if val[3] is None else val[3]) for val in batch])

        q_s_a = self._model.predict_batch(states, self._sess)  # predict Q(s,a) given the batch of states
        q_s_a_d = self._model.predict_batch(next_states, self._sess)  # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below

        x = np.zeros((len(batch), self._model.num_states))
        y = np.zeros((len(batch), self._model.num_actions))

        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            current_q = q_s_a[i]
            if next_state is None:
                current_q[action] = reward
            else:
                current_q[action] = reward + self._GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def performance_store(self):
        return self._performance_store


def train(params):

    active_days_file = params['active_days_file']
    if not os.path.isfile(active_days_file):
        print(colored("ERROR: " + active_days_file + " not found!", color="red"))
    else:
        movers = pd.read_csv(active_days_file)

        tr = Trade_Env(movers, sim_chart_index=None, simulation_mode=False)

        print(tr.num_actions, tr.num_states)

        model = Model(tr.num_states, tr.num_actions, params)
        mem = Memory(params['MEMORY'])

        # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        with tf.Session() as sess:
            model.restore(sess)

            bot = TrainerBot(sess, model, tr, mem, params, False)
            num_episodes = params['EPISODES']
            cnt = 0
            while cnt < num_episodes:
                print('Episode {} of {}'.format(cnt+1, num_episodes))
                bot.run()
                cnt += 1
                if cnt % params['CHECKPOINT_AT_EPISODE_STEP'] == 0:
                    model.save(sess, cnt)

            print(bot.performance_store)
            print(bot.reward_store)
            plt.plot(bot.performance_store)
            plt.show()
            plt.plot(bot.reward_store)
            plt.show()

            plt.close("all")


def get_params():
    params = {
        'MAX_EPSILON': 0.12,
        'MIN_EPSILON': 0.1,
        'LAMBDA': 0.00001,
        'GAMMA': 0.99,

        'BATCH_SIZE': 2000,
        'TRAINING_START': 10000,
        'MEMORY': 100000,

        'EPISODES': 10000,
        'CHECKPOINT_AT_EPISODE_STEP': 500,
        'MAX_CHECKPOINTS': 50,

        'STATS_PER_STEP': 50,
        'active_days_file': "data\\active_days.csv"
    }

    return params


if __name__ == "__main__":
    train(get_params())
